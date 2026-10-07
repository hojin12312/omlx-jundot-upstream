#!/usr/bin/env python3
"""Q8A8 quality gate: the same Q8 checkpoint with A16 activations (reference) against A8 activations.

The model is loaded by oMLX's own VLM engine start-up (qwen35_oq_a8_enabled on). Per sequence one chunk-wise
prefill (production chunk of 2048 rows) is run three times with the same weights and inputs, one cache each:

  ref   OMLX_OQ_A8=0 (every oQ A8 path declines: stock Q8 A16)
  cand  A8 on (Q8A8)
  ctrl  OMLX_OQ_A8=0 again (harness noise: must be exactly 0)

Metrics per 128-row block: KL(ref||cand) on float32 full-vocabulary logits, top-1 agreement, NLL of the true next
token under each pass, NaN/Inf. The reference is the SAME checkpoint with A16 activations, so the weight
quantization error of Q8 is not part of the comparison. Crash safe: one fsynced JSON line per sequence.
"""
import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import mlx.core as mx

BLOCK = 128
# Per-call production switch: OMLX_OQ_A8=0 makes every oQ A8 path decline (stock Q8 A16), unset runs A8.
PASSES = (
    ("ref", {"OMLX_OQ_A8": "0"}),
    ("cand", {"OMLX_OQ_A8": None}),
    ("ctrl", {"OMLX_OQ_A8": "0"}),
)
COMPARISONS = (("a8", "ref", "cand"), ("ctrl", "ref", "ctrl"))


def block_metrics(la, lb, targets):
    a, b = la.astype(mx.float32), lb.astype(mx.float32)
    lpa, lpb = a - mx.logsumexp(a, axis=-1, keepdims=True), b - mx.logsumexp(b, axis=-1, keepdims=True)
    kl = mx.sum(mx.exp(lpa) * (lpa - lpb), axis=-1)
    agree = mx.argmax(lpa, axis=-1) == mx.argmax(lpb, axis=-1)
    t = mx.maximum(targets, 0)[:, None]
    nll_a = -mx.take_along_axis(lpa, t, axis=-1)[:, 0]
    nll_b = -mx.take_along_axis(lpb, t, axis=-1)[:, 0]
    mx.eval(kl, agree, nll_a, nll_b)
    return np.array(kl), np.array(agree), np.array(nll_a), np.array(nll_b)


def finite(x):
    ok = mx.all(mx.isfinite(x.astype(mx.float32)))
    mx.eval(ok)
    return bool(ok.item())


def weighted_ci(rows, key_a, key_b):
    d = np.array([r[key_b] - r[key_a] for r in rows])
    w = np.array([r["tokens"] for r in rows], dtype=np.float64)
    mean = float((d * w).sum() / w.sum())
    se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None
    return mean, se, d


async def run(args):
    import omlx
    from omlx.custom_kernels.qwen35_prefill import _ext
    tree = args.tree.rstrip("/")
    assert omlx.__file__.startswith(tree + "/") and _ext.__file__.startswith(tree + "/"), omlx.__file__
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.model_settings import ModelSettings
    from omlx.custom_kernels.qwen35_prefill import fast
    from omlx.patches import qwen35_oq_a8 as oa
    from omlx.scheduler import SchedulerConfig

    now = {"v": "ref"}
    cov = {n: defaultdict(int) for n, _ in PASSES}

    real_apply = oa.apply_plan

    def counting_apply(linear, stage, plan):
        c = cov[now["v"]]
        c["a8_proj"] += 1
        c[f"a8_q{plan.bits}"] += 1
        return real_apply(linear, stage, plan)

    real_stage = oa.stage_a

    def counting_stage(*a, **k):
        cov[now["v"]]["stage_a"] += 1
        return real_stage(*a, **k)

    real_native = fast.qwen35_q8_affine_qmm_t

    def counting_native(*a, **k):
        cov[now["v"]]["q8_a16_native"] += 1
        return real_native(*a, **k)

    oa.apply_plan = counting_apply
    oa.stage_a = counting_stage
    fast.qwen35_q8_affine_qmm_t = counting_native

    def setmode(name):
        now["v"] = name
        for k, v in dict(PASSES)[name].items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    setmode("cand")
    ms = ModelSettings(qwen35_oq_a8_enabled=True, qwen35_oq_a8_min_tokens=128)
    mx.set_wired_limit(64 * 1024 ** 3)
    eng = VLMBatchedEngine(model_name=args.ckpt, model_settings=ms, scheduler_config=SchedulerConfig())
    t0 = time.time()
    await eng.start()
    print(f"engine up in {time.time() - t0:.0f}s", flush=True)
    lm = eng._vlm_model.language_model
    full = json.load(open(args.corpus))["sequences"]
    picks = [int(i) for i in args.pick.split(",")] if args.pick != "all" else list(range(len(full)))
    out_path = Path(args.out)
    partial = Path(str(out_path.with_suffix("")) + ".partial.jsonl")
    ppdir = Path(str(out_path.with_suffix("")) + "_pp")
    ppdir.mkdir(parents=True, exist_ok=True)
    done = {}
    if partial.exists():
        for line in partial.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if (ppdir / f"s{r['index']}.npz").exists():
                done[r["index"]] = r
    if done:
        print(f"resuming: {len(done)} sequences already done: {sorted(done)}", flush=True)
    results = list(done.values())
    names = [n for n, _ in PASSES]
    for si in picks:
        seq = full[si]
        if si in done:
            continue
        toks = mx.array(seq["tokens"], dtype=mx.int32)[None, :]
        n = toks.shape[1]
        caches = {k: lm.make_cache() for k in names}
        cov_before = {m: dict(v) for m, v in cov.items()}
        acc = {c: {"kl": [], "agree": [], "na": [], "nb": []} for c, _, _ in COMPARISONS}
        finite_all = {k: True for k in names}
        last = {}
        t1 = time.time()
        for start in range(0, n, args.chunk):
            end = min(n, start + args.chunk)
            chunk = toks[:, start:end]
            outs = {}
            for name in names:
                setmode(name)
                o = lm(chunk, cache=caches[name])
                lg = o.logits[0]
                mx.eval(lg)
                finite_all[name] &= finite(lg)
                outs[name] = lg
                last[name] = lg[-1]
            tgt = np.full(end - start, -1, dtype=np.int32)
            nxt = np.array(toks[0, start + 1: min(n, end + 1)])
            tgt[: len(nxt)] = nxt
            for r in range(0, end - start, BLOCK):
                sl = slice(r, min(end - start, r + BLOCK))
                t = mx.array(tgt[sl])
                valid = tgt[sl] >= 0
                for cname, first, second in COMPARISONS:
                    kl, ag, na, nb = block_metrics(outs[first][sl], outs[second][sl], t)
                    a = acc[cname]
                    a["kl"].append(kl)
                    a["agree"].append(ag)
                    a["na"].append(np.where(valid, na, np.nan))
                    a["nb"].append(np.where(valid, nb, np.nan))
            del outs
            mx.clear_cache()
        gen = {}
        for name in ("ref", "cand"):
            setmode(name)
            tok = mx.argmax(last[name], axis=-1)[None]
            seqg = []
            for _ in range(args.decode):
                seqg.append(int(tok.item()))
                o = lm(tok[None, :].astype(mx.int32), cache=caches[name])
                lg = o.logits if getattr(o, "logits", None) is not None else o
                tok = mx.argmax(lg[0, -1, :], axis=-1)[None]
            gen[name] = seqg
        first_div = next((i for i, (p, q) in enumerate(zip(gen["ref"], gen["cand"])) if p != q), None)

        rec = {"index": si, "name": seq["name"], "category": seq["category"], "tokens": int(n),
               "finite": finite_all, "greedy": gen, "greedy_div": first_div,
               "seconds": time.time() - t1,
               "coverage_delta": {m: {k: cov[m][k] - cov_before[m].get(k, 0) for k in cov[m]} for m in cov}}
        pp = {}
        for cname, first, second in COMPARISONS:
            a = acc[cname]
            kl, ag = np.concatenate(a["kl"]), np.concatenate(a["agree"])
            na, nb = np.concatenate(a["na"]), np.concatenate(a["nb"])
            ok = ~np.isnan(na)
            dpos = (nb - na)[ok]
            rec[cname] = {"kl_mean": float(kl.mean()), "kl_p95": float(np.percentile(kl, 95)),
                          "kl_p99": float(np.percentile(kl, 99)), "kl_max": float(kl.max()),
                          "top1_agree": float(ag.mean()),
                          "nll_first": float(na[ok].mean()), "nll_second": float(nb[ok].mean()),
                          "nll_delta": float(nb[ok].mean() - na[ok].mean()),
                          "dnll_pos_p01": float(np.percentile(dpos, 1)), "dnll_pos_p99": float(np.percentile(dpos, 99)),
                          "dnll_pos_min": float(dpos.min()), "dnll_pos_max": float(dpos.max())}
            pp[f"{cname}_kl"] = kl.astype(np.float32)
            pp[f"{cname}_dnll"] = np.where(ok, nb - na, np.nan).astype(np.float32)
            pp[f"{cname}_agree"] = ag
        np.savez_compressed(ppdir / f"s{si}.npz", **pp)
        with open(partial, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        results.append(rec)
        cd = rec["coverage_delta"]
        print(f"[{len(results)}/{len(picks)}] {seq['category']:<10} {seq['name'][:28]:<28} n={n} "
              f"dNLL {rec['a8']['nll_delta']:+.5f} KL {rec['a8']['kl_mean']:.2e} top1 {rec['a8']['top1_agree']:.4f} "
              f"| ctrl KLmax {rec['ctrl']['kl_max']:.1e} finite {all(finite_all.values())} "
              f"| a8_proj cand {cd['cand'].get('a8_proj', 0)} ref {cd['ref'].get('a8_proj', 0)} "
              f"native {cd['cand'].get('q8_a16_native', 0)}", flush=True)
        del caches
        mx.clear_cache()
    results.sort(key=lambda r: r["index"])
    toks_total = sum(r["tokens"] for r in results)
    summary = {"sequences": len(results), "positions": toks_total, "chunk": args.chunk}
    for cname, first, second in COMPARISONS:
        rows = [{"tokens": r["tokens"], "a": r[cname]["nll_first"], "b": r[cname]["nll_second"]} for r in results]
        mean, se, d = weighted_ci(rows, "a", "b")
        pps = [np.load(ppdir / f"s{r['index']}.npz") for r in results]
        allkl = np.concatenate([z[f"{cname}_kl"] for z in pps])
        allag = np.concatenate([z[f"{cname}_agree"] for z in pps])
        summary[cname] = {
            "pair": [first, second],
            "nll_first_weighted": float(sum(r[cname]["nll_first"] * r["tokens"] for r in results) / toks_total),
            "nll_second_weighted": float(sum(r[cname]["nll_second"] * r["tokens"] for r in results) / toks_total),
            "nll_delta_weighted": mean, "se_over_sequences": se,
            "ci95": [mean - 1.96 * se, mean + 1.96 * se] if se is not None else None,
            "ppl_ratio": float(np.exp(mean)),
            "kl_mean_weighted": float(sum(r[cname]["kl_mean"] * r["tokens"] for r in results) / toks_total),
            "kl_p95_all": float(np.percentile(allkl, 95)), "kl_p99_all": float(np.percentile(allkl, 99)),
            "kl_max_all": float(allkl.max()),
            "top1_agree_weighted": float(sum(r[cname]["top1_agree"] * r["tokens"] for r in results) / toks_total),
            "top1_agree_all_positions": float(allag.mean()),
            "positive_delta_sequences": int((d > 0).sum())}
    summary["control_kl_max"] = max(r["ctrl"]["kl_max"] for r in results)
    summary["all_finite"] = all(all(r["finite"].values()) for r in results)
    summary["greedy_identical"] = sum(r["greedy_div"] is None for r in results)
    summary["coverage_by_pass"] = {m: {k: sum(r["coverage_delta"][m].get(k, 0) for r in results)
                                       for k in {kk for r in results for kk in r["coverage_delta"][m]}}
                                   for m, _ in PASSES}
    print(json.dumps(summary, indent=2))
    out_path.write_text(json.dumps({"summary": summary, "sequences": results, "tree": tree,
                                    "omlx_file": omlx.__file__}, indent=2))
    try:
        await eng.stop()
    except Exception:  # noqa: BLE001
        pass
    time.sleep(3.5)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--pick", default="all", help="comma list of sequence indices, or all")
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--decode", type=int, default=24)
    ap.add_argument("--out", "-o", required=True)
    asyncio.run(run(ap.parse_args()))
