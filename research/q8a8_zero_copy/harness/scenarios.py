#!/usr/bin/env python3
"""Prefix-cache, chunk-size and reload scenarios against one server process (research harness).

modes
  cache   miss, identical hit, partial hit, new-suffix hit, then unload + SSD cache wipe + reload and a request again
  chunks  the same prompts at several prefill chunk sizes (the scheduler base size is overridden through the hook)
Every request is a streamed temperature-0 completion. Recorded per request: TTFT, prompt and cached token counts,
Q8 A8 rows per call (histogram), native/transposed call counts, prepared-copy scan, MLX memory, and the text.
"""
import argparse, hashlib, json, os, shutil, sys, time, urllib.request
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
import pp2_helpers as m  # noqa: E402
from common import Ctl, sh  # noqa: E402

HOOK_DIR = HARNESS / "hook"
GiB = 1024 ** 3


def main():  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--mode", choices=["cache", "chunks", "toggle"], required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--a8", choices=["on", "off"], default="on")
    ap.add_argument("--mtp", choices=["on", "off"], default="off")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096])
    ap.add_argument("--chunks", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--blocks", nargs="+", default=["native", "transposed", "transposed", "native"],
                    help="toggle mode: the arm order, one block per entry")
    ap.add_argument("--per-block", type=int, default=3)
    ap.add_argument("--subset", choices=["all", "mlp", "rest"], default=None,
                    help="toggle mode: switch only this projection family (the rest stays group-major)")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--settle", type=float, default=8.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for k in [k for k in os.environ if k.startswith("OMLX_") or k == "PYTHONPATH"]:
        del os.environ[k]
    tag = f"{args.mode}_{args.arm}"
    base = Path(args.workdir) / f"scn_{tag}"
    shutil.rmtree(base, ignore_errors=True)
    (base / "logs").mkdir(parents=True)
    (base / "home").mkdir()
    ssd = base / "ssd"
    ssd.mkdir()
    cache_on = args.mode == "cache"
    settings = {"version": 1.0, "server": {"host": "127.0.0.1", "log_level": "info"},
                "model": {"model_dir": args.model_dir, "model_fallback": False},
                "cache": ({"enabled": True, "ssd_cache_dir": str(ssd), "ssd_cache_max_size": "40GB"} if cache_on
                          else {"enabled": False}),
                "scheduler": {"max_concurrent_requests": 1, "chunked_prefill": True},
                "sampling": {"temperature": 0.0, "max_tokens": 4096},
                "auth": {"api_key": None, "skip_api_key_verification": True},
                "logging": {"log_dir": str(base / "logs"), "retention_days": 1},
                "memory": {"prefill_memory_guard": False}, "huggingface": {"hf_cache_enabled": False}}
    m.write_settings(base, settings)
    ms = {"qwen35_oq_a8_enabled": args.a8 == "on", "qwen35_oq_a8_min_tokens": 128, "mtp_enabled": args.mtp == "on"}
    (base / "model_settings.json").write_text(json.dumps({"version": 1, "models": {args.model: ms}}, indent=2))
    ctl = Ctl(base / "ctl")
    env = {"PYTHONPATH": args.tree + ":" + str(HOOK_DIR), "HOME": str(base / "home"), "OMLX_DEC_CTL": str(base / "ctl")}
    for kv in args.env:
        k, v = kv.split("=", 1)
        env[k] = v

    tok = m.load_tokenizer(Path(args.checkpoint))
    flat = json.load(open(args.corpus))["tokens"]
    res = {"schema": 1, "mode": args.mode, "arm": args.arm, "argv": sys.argv, "tree": args.tree,
           "tree_git_head": sh("git", "-C", args.tree, "rev-parse", "HEAD"), "model_settings": ms, "env": env,
           "requests": [], "events": []}
    out_path = Path(args.out)

    def save():
        out_path.write_text(json.dumps(res, indent=1))

    def ids_for(nonce, length, skew=256):
        off = int(hashlib.sha256(nonce.encode()).hexdigest(), 16) % len(flat)
        return (flat[off:] + flat[:off])[:length + skew]

    def exact_prompt(seq, length):
        cut, seen = length, set()
        for _ in range(40):
            prompt = tok.decode(seq[:cut])
            n = len(tok.encode(prompt).ids)
            if n == length or cut in seen:
                break
            seen.add(cut)
            cut += length - n
        return prompt, n

    port = m.free_port(8450)
    url = f"http://127.0.0.1:{port}"
    log_path = base / "logs" / "server.log"
    proc = m.start_server(args.python, base, Path(args.model_dir), port, env, log_path, [])

    def post(path, body=b"", timeout=900):
        req = urllib.request.Request(url + path, data=body, method="POST", headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=timeout).read()

    def request(label, prompt, planned, max_tokens=None):
        mt = max_tokens or args.max_tokens
        body = m.request_body(args.model, prompt, mt)
        time.sleep(args.settle)
        ctl.call("counters", reset=True)
        mem0 = ctl.call("mem", reset_peak=True)
        req = urllib.request.Request(url + "/v1/completions", data=body, method="POST", headers={
            "Content-Type": "application/json", "Accept": "text/event-stream"})
        t0 = time.perf_counter()
        events, usage, text, finish = [], {}, "", None
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                d = line[5:].strip()
                if d == "[DONE]":
                    break
                try:
                    obj = json.loads(d)
                except json.JSONDecodeError:
                    continue
                now = time.perf_counter() - t0
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    if ch.get("text"):
                        events.append(now)
                        text += ch["text"]
        total = time.perf_counter() - t0
        time.sleep(1.0)
        mem1 = ctl.call("mem")
        cnt = ctl.call("counters")
        time.sleep(8.0)
        mem2 = ctl.call("mem")
        scan = ctl.call("scan_prepared")
        rec = {"label": label, "planned_prompt_ids": planned, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()[:16],
               "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "ttft_s": events[0] if events else None, "total_s": total, "finish": finish,
               "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
               "counts": cnt["counts"], "rows_hist": cnt.get("rows_hist"),
               "mem_before": mem0, "mem_after": mem1, "mem_settled": mem2,
               "prepared_scan": {k: scan.get(k) for k in ("layers_with_prepared_copy", "prepared_copy_bytes")}}
        res["requests"].append(rec)
        save()
        a8 = cnt["counts"]
        print(f"[{tag}] {label:<28} ptok={rec['prompt_tokens']} cached={rec['cached_tokens']} "
              f"ttft={rec['ttft_s'] and round(rec['ttft_s'], 3)} nat={a8.get('a8_q8_native', 0)} "
              f"tr={a8.get('a8_q8_transposed', 0)} peak={mem1['peak'] / GiB:.2f} act={mem2['active'] / GiB:.2f} "
              f"prep={scan.get('layers_with_prepared_copy')} rows={rec['rows_hist']}", flush=True)
        return rec

    def wait_wired():
        for _ in range(120):
            fl = ctl.call("flags")
            if args.a8 == "off" or (fl["mlp_patched"] and fl["gdn_registered"]):
                return fl
            time.sleep(0.5)
        raise SystemExit("A8 patch not wired")

    try:
        m.wait_ready(url + "/v1/models", proc)
        p32, _ = exact_prompt(ids_for("load", 64), 32)
        post("/v1/completions", m.request_body(args.model, p32, 1))
        res["flags"] = wait_wired()
        assert ctl.call("install")["installed"] or args.a8 == "off"
        res["mem_after_load"] = ctl.call("mem", reset_peak=True)

        if args.mode == "cache":
            for L in args.lengths:
                seq = ids_for(f"cache-{L}", 3 * L)
                tail = ids_for(f"cache-tail-{L}", 3 * L)
                p1, n1 = exact_prompt(seq, L)
                half = L // 2
                p_part, np_ = exact_prompt(seq[:half] + tail, L)          # shares the first half, new second half
                p_ext, ne = exact_prompt(seq[:L] + tail[:L // 4], L + L // 4)  # P1 plus a new suffix
                p_s100, n100 = exact_prompt(seq[:L] + tail[:100], L + 100)   # uncached rows below the A8 floor
                p_s130, n130 = exact_prompt(seq[:L] + tail[:130], L + 130)   # unaligned, just above the floor
                request(f"L{L}.miss", p1, n1)
                request(f"L{L}.hit_identical", p1, n1)
                request(f"L{L}.partial_half", p_part, np_)
                request(f"L{L}.hit_prefix+suffix", p_ext, ne)
                request(f"L{L}.hit_suffix100", p_s100, n100)
                request(f"L{L}.hit_suffix130", p_s130, n130)
                # invalidation: unload, wipe the SSD tier, load again
                post(f"/v1/models/{args.model}/unload")
                time.sleep(3.0)
                res["events"].append({"unloaded": ctl.call("mem")})
                for child in ssd.iterdir():
                    shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink()
                post(f"/v1/models/{args.model}/load")
                time.sleep(3.0)
                res["flags_after_reload"] = wait_wired()
                assert ctl.call("install")["installed"] or args.a8 == "off"
                request(f"L{L}.after_wipe_reload", p1, n1)
                request(f"L{L}.hit_after_reload", p1, n1)
        elif args.mode == "toggle":
            for L in args.lengths:
                if args.subset:
                    pw0, nw0 = exact_prompt(ids_for(f"toggle-warm0-{L}", L), L)
                    request(f"L{L}.init", pw0, nw0)          # creates the plans before they are edited
                for bi, arm in enumerate(args.blocks):
                    if args.subset:
                        ack = ctl.call("set_plans", native=(arm == "native"), subset=args.subset)
                    else:
                        ack = ctl.call("set_native", native=(arm == "native"))
                    res["events"].append({"block": bi, "arm": arm, "length": L, "switch": ack})
                    pw, nw = exact_prompt(ids_for(f"toggle-warm-{L}", L), L)
                    request(f"L{L}.b{bi}.{arm}.warm", pw, nw)
                    for k in range(args.per_block):
                        p, npl = exact_prompt(ids_for(f"toggle-{L}-{k}", L), L)
                        request(f"L{L}.b{bi}.{arm}.k{k}", p, npl)
        else:
            for L in args.lengths:
                for extra in (0, 3):                     # aligned and unaligned token counts
                    n_tok = L + extra
                    p, npl = exact_prompt(ids_for(f"chunk-{n_tok}", n_tok), n_tok)
                    for c in args.chunks:
                        ctl.call("set_chunk", size=c)
                        request(f"L{n_tok}.chunk{c}", p, npl)
                    ctl.call("set_chunk", size=0)
                    request(f"L{n_tok}.chunk_default", p, npl)
        res["log_suspect_lines"] = [l[:300] for l in log_path.read_text(errors="replace").splitlines()
                                    if any(w in l for w in ("Traceback", "ERROR", "WARNING", "fallback", "oq_a8"))][:60]
    finally:
        m.stop_server(proc)
        save()
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
