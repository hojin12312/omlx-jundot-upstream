#!/usr/bin/env python3
"""Iteration 3 long-context memory: B0 (#4320: Gate+Up A8, Down A16) vs B1 (production: + Down A8).

One process, the production tree. For every (length, condition) a real chunked prefill is run on a
fresh cache and the MLX allocator is read before, at peak and after: active, peak, cache, and the
settled value after ``mx.clear_cache()``. Conditions alternate B0 B1 B1 B0 (--rounds times) after one
warm-up per condition. The toggle is the production variable ``OMLX_M5_ROUTED_DOWN_A8``.

Persistent overhead = settled active memory after the prefill cache is released (B1 minus B0) and
the active memory before the first prefill of each condition (a persistent copy would show there).

Safety (the machine has 128 GB and the model alone is ~98 GiB): the wired limit is set like the oMLX
server does, lengths run in increasing order, and before each length the peak is projected linearly
from the last two measured lengths; when the projection exceeds --max-peak-gib the length is not run
and is recorded as infeasible.
"""
import argparse
import asyncio
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx

GiB = 1024 ** 3


def snap():
    return {"active": mx.get_active_memory() / GiB, "cache": mx.get_cache_memory() / GiB,
            "peak": mx.get_peak_memory() / GiB}


async def run(args):
    import omlx
    from omlx.custom_kernels.qwen35_prefill import _ext
    tree = args.tree.rstrip("/")
    assert omlx.__file__.startswith(tree + "/") and _ext.__file__.startswith(tree + "/"), omlx.__file__
    from omlx.engine.vlm import VLMBatchedEngine
    from omlx.model_settings import ModelSettings
    from omlx.patches import m5_gather_qmm_a8 as a8
    from omlx.scheduler import SchedulerConfig

    assert hasattr(a8, "_try_down_a8") and not hasattr(a8, "_research_down_on"), "not the production tree"
    cov = {"v": "0", "0": defaultdict(int), "1": defaultdict(int)}
    real = a8.sorted_gather_qmm_a8

    def counting(*a, **k):
        out = real(*a, **k)
        if out is not None:
            cov[cov["v"]]["gate_up" if k.get("swiglu") else "down"] += 1
        return out

    a8.sorted_gather_qmm_a8 = counting

    def setmode(c):
        os.environ["OMLX_M5_ROUTED_DOWN_A8"] = c
        cov["v"] = c

    setmode("0")
    ms = ModelSettings(qwen35_oq_a8_enabled=True, qwen35_oq_a8_min_tokens=128)
    # Keep the weights resident like the oMLX server does (see quality_prod.py for the incident).
    mx.set_wired_limit(112 * GiB)
    eng = VLMBatchedEngine(model_name=args.ckpt, model_settings=ms, scheduler_config=SchedulerConfig())
    await eng.start()
    lm = eng._vlm_model.language_model
    corpus_doc = json.load(open(args.corpus))
    base = corpus_doc["tokens"] if "tokens" in corpus_doc else [t for s in corpus_doc["sequences"] for t in s["tokens"]]
    import hashlib
    import subprocess
    import sys
    import importlib.metadata as md
    out = {"tree": tree, "chunk": args.chunk, "rows": [], "idle_before": {}, "infeasible": {}, "projection": {},
           "tree_git_head": subprocess.run(["git", "-C", tree, "rev-parse", "HEAD"], capture_output=True,
                                           text=True).stdout.strip(),
           "a8_source_sha256": hashlib.sha256(open(a8.__file__, "rb").read()).hexdigest(),
           "native_ext_sha256": hashlib.sha256(open(_ext.__file__, "rb").read()).hexdigest(),
           "python": sys.version.split()[0], "mlx": md.version("mlx"),
           "corpus_sha256": hashlib.sha256(json.dumps(base, separators=(",", ":")).encode()).hexdigest(),
           "argv": sys.argv}
    mx.clear_cache()
    out["idle_before"]["start"] = snap()

    def prefill(length, salt, cond):
        setmode(cond)
        off = (salt * 7919) % max(1, len(base) - length)
        toks = mx.array(base[off: off + length], dtype=mx.int32)[None, :]
        mx.clear_cache()
        pre = snap()
        mx.reset_peak_memory()
        cache = lm.make_cache()
        c0 = dict(cov[cond])
        t0 = time.time()
        for s in range(0, length, args.chunk):
            o = lm(toks[:, s: s + args.chunk], cache=cache)
            mx.eval(o.logits)
            del o
        secs = time.time() - t0
        at_end = snap()
        calls = {k: cov[cond][k] - c0.get(k, 0) for k in ("gate_up", "down")}
        del cache
        settled_cache = snap()
        mx.clear_cache()
        settled = snap()
        return {"pre": pre, "at_end": at_end, "after_cache_release": settled_cache, "settled": settled,
                "calls": calls, "seconds": secs}

    salt = 0
    peaks = []
    for length in sorted(args.lengths):
        if len(peaks) >= 2:
            (l0, p0), (l1, p1) = peaks[-2], peaks[-1]
            proj = p1 + (p1 - p0) / (l1 - l0) * (length - l1)
            out["projection"][str(length)] = proj
            print(f"projected peak at {length}: {proj:.2f} GiB (limit {args.max_peak_gib})", flush=True)
            if proj > args.max_peak_gib:
                out["infeasible"][str(length)] = {"projected_peak_gib": proj, "limit_gib": args.max_peak_gib}
                print(f"L={length}: NOT RUN, projected peak above the limit", flush=True)
                continue
        for cond in ("0", "1"):
            prefill(length, 1000 + salt, cond)  # warm-up (compiles kernels, sizes the allocator)
            salt += 1
        order = []
        for _ in range(args.rounds):
            order += ["0", "1", "1", "0"]
        for cond in order:
            salt += 1
            r = prefill(length, salt, cond)
            r.update({"length": length, "cond": cond})
            out["rows"].append(r)
            print(f"L={length:>6} B{cond} peak={r['at_end']['peak']:.3f} GiB active_end={r['at_end']['active']:.3f} "
                  f"cache_end={r['at_end']['cache']:.3f} settled={r['settled']['active']:.3f} "
                  f"gate_up={r['calls']['gate_up']} down={r['calls']['down']} {r['seconds']:.0f}s", flush=True)
            Path(args.out).write_text(json.dumps(out, indent=1))
        peaks.append((length, max(r["at_end"]["peak"] for r in out["rows"] if r["length"] == length)))
    Path(args.out).write_text(json.dumps(out, indent=1))
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
    ap.add_argument("--lengths", type=int, nargs="+", default=[16384, 32768, 65536, 131072])
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--max-peak-gib", type=float, default=111.0)
    ap.add_argument("--out", "-o", required=True)
    asyncio.run(run(ap.parse_args()))
