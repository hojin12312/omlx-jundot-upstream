#!/usr/bin/env python3
"""E2 concurrency and stability session (one server process, concurrency 2).

Scenarios, each followed by a quiet period and a memory/lifetime sample:
  pp   two 8192-token prefills submitted together
  pd   a 512-token request that is decoding 128 tokens while an 8192-token prefill arrives
  cancel  an 8192-token request whose client disconnects 1.5 s into the prefill, then a normal request
  stagger a short request finishing while a long one is still prefilling
  loop    N sequential requests (alternating 512 / 2048 tokens, 32 new tokens) with memory sampled after each

Reported per scenario: client timings, completion tokens, server-side lifetime stats (holders,
prepare batches, drops), counters, active / cache / phys_footprint. Stops at the first request error.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pp2_helpers as m  # noqa: E402
from common import Ctl, sh  # noqa: E402

HOOK_DIR = HERE / "hook"
GiB = 1024 ** 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--loop-requests", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=2, help="max_concurrent_requests (1, 2 or 4)")
    ap.add_argument("--cache", action="store_true", help="prefix cache on (SSD tier in the session directory)")
    ap.add_argument("--mtp", choices=["on", "off"], default="off")
    ap.add_argument("--extra", action="store_true", help="C4 prefill smoke, cancel with prefix cache, unload/reload")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    for k in [k for k in os.environ if k.startswith("OMLX_") or k == "PYTHONPATH"]:
        del os.environ[k]
    base = Path(args.workdir) / f"conc_{args.arm}"
    shutil.rmtree(base, ignore_errors=True)
    (base / "logs").mkdir(parents=True)
    (base / "home").mkdir()
    settings = {"version": 1.0, "server": {"host": "127.0.0.1", "log_level": "info"},
                "model": {"model_dir": args.model_dir, "model_fallback": False}, "cache": ({"enabled": True, "ssd_cache_dir": str(base / "ssd"), "ssd_cache_max_size": "40GB"} if args.cache
                          else {"enabled": False}),
                "scheduler": {"max_concurrent_requests": args.concurrency, "chunked_prefill": True},
                "sampling": {"temperature": 0.0, "max_tokens": 4096},
                "auth": {"api_key": None, "skip_api_key_verification": True},
                "logging": {"log_dir": str(base / "logs"), "retention_days": 1},
                "memory": {"prefill_memory_guard": False}, "huggingface": {"hf_cache_enabled": False}}
    m.write_settings(base, settings)
    (base / "ssd").mkdir(exist_ok=True)
    ms = {"qwen35_oq_a8_enabled": True, "qwen35_oq_a8_min_tokens": 128, "mtp_enabled": args.mtp == "on"}
    (base / "model_settings.json").write_text(json.dumps({"version": 1, "models": {args.model: ms}}, indent=2))
    ctl = Ctl(base / "ctl")
    env = {"PYTHONPATH": args.tree + ":" + str(HOOK_DIR), "HOME": str(base / "home"), "OMLX_DEC_CTL": str(base / "ctl")}
    for kv in args.env:
        k, v = kv.split("=", 1)
        env[k] = v

    tok = m.load_tokenizer(Path(args.checkpoint))
    flat = json.load(open(args.corpus))["tokens"]

    def build(length, nonce):
        salt_ids = tok.encode(f"[conc {nonce}] ").ids
        off = int(hashlib.sha256(nonce.encode()).hexdigest(), 16) % len(flat)
        seq = salt_ids + (flat[off:] + flat[:off])[:length + 256 - len(salt_ids)]
        cut, seen = length, set()
        for _ in range(40):
            prompt = tok.decode(seq[:cut])
            n = len(tok.encode(prompt).ids)
            if n == length or cut in seen:
                break
            seen.add(cut)
            cut += length - n
        return prompt

    port = m.free_port(8450)
    url = f"http://127.0.0.1:{port}"
    log_path = base / "logs" / "server.log"
    proc = m.start_server(args.python, base, Path(args.model_dir), port, env, log_path, [])
    res = {"schema": 1, "arm": args.arm, "env": env, "tree": args.tree,
           "tree_git_head": sh("git", "-C", args.tree, "rev-parse", "HEAD"), "scenarios": []}

    def save():
        Path(args.out).write_text(json.dumps(res, indent=1))

    def request(length, nonce, max_tokens, cancel_after=None):
        """Streaming completion over http.client; ``cancel_after`` closes the socket from a timer
        (independent of whether any bytes have arrived, so it lands inside the prefill)."""
        import http.client

        body = m.request_body(args.model, build(length, nonce), max_tokens)
        rec = {"length": length, "nonce": nonce, "max_tokens": max_tokens, "t_submit": time.perf_counter()}
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
        timer = None
        first, usage, text = None, {}, ""
        t0 = time.perf_counter()
        try:
            conn.request("POST", "/v1/completions", body=body,
                         headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
            if cancel_after is not None:
                def kill():
                    rec["cancelled"] = True
                    try:
                        conn.sock.shutdown(2)
                        conn.sock.close()
                    except Exception:  # noqa: BLE001
                        pass
                timer = threading.Timer(cancel_after, kill)
                timer.start()
            resp = conn.getresponse()
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    if ch.get("text"):
                        first = first if first is not None else time.perf_counter() - t0
                        text += ch["text"]
        except Exception as exc:  # noqa: BLE001
            rec["error"] = repr(exc)
        finally:
            if timer is not None:
                timer.cancel()
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        rec.update(ttft=first, total=time.perf_counter() - t0, completion_tokens=usage.get("completion_tokens"),
                   prompt_tokens=usage.get("prompt_tokens"), text_sha=hashlib.sha256(text.encode()).hexdigest()[:16])
        return rec

    def sample(label, wait=0.0):
        time.sleep(wait)
        mem = ctl.call("mem")
        return {"label": label, "mem": mem, "counters": ctl.call("counters")["counts"],
                "lifetime": ctl.call("lifetime")["stats"]}

    def scenario(name, jobs):
        """jobs: list of (delay_s, kwargs). Runs them in threads, then samples after 16 s of quiet."""
        ctl.call("counters", reset=True)
        ctl.call("mem", reset_peak=True)
        out = [None] * len(jobs)

        def run(i, delay, kw):
            time.sleep(delay)
            out[i] = request(**kw)

        ths = [threading.Thread(target=run, args=(i, d, kw)) for i, (d, kw) in enumerate(jobs)]
        t0 = time.perf_counter()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.perf_counter() - t0
        sm_end = sample("end")
        sm_quiet = sample("quiet16s", wait=16.0)
        rec = {"name": name, "wall": wall, "requests": out, "end": sm_end, "quiet": sm_quiet,
               "clear_events": ctl.call("counters")["clear_events"]}
        res["scenarios"].append(rec)
        save()
        bad = [r for r in out if r.get("error") and not r.get("cancelled")]
        print(f"[{args.arm}] {name}: wall={wall:.1f}s peak={sm_end['mem']['peak']/GiB:.2f} "
              f"quiet_active={sm_quiet['mem']['active']/GiB:.2f} phys={sm_quiet['mem']['phys_footprint']/GiB:.2f} "
              f"holders={(sm_quiet['lifetime'] or {}).get('holders')} errors={len(bad)}", flush=True)
        if bad:
            raise SystemExit(f"stopping: request error in {name}: {bad[0]['error']}")
        return rec

    try:
        m.wait_ready(url + "/v1/models", proc)
        request(32, "load", 1)
        for _ in range(60):
            fl = ctl.call("flags")
            if fl["mlp_patched"] and fl["gdn_registered"]:
                break
            time.sleep(0.5)
        assert fl["oq_a8_available"] is True and fl["omlx_file"].startswith(args.tree + "/"), fl
        assert ctl.call("install")["installed"]
        res["flags"] = fl
        # warm every route once so the first discovery build is not part of a scenario
        request(2048, "warm-a", 8)
        time.sleep(10)
        res["baseline_sample"] = sample("after_warm", wait=10.0)

        scenario("pp_two_8192_prefills", [(0.0, dict(length=8192, nonce="pp-a", max_tokens=16)),
                                           (0.1, dict(length=8192, nonce="pp-b", max_tokens=16))])
        scenario("pd_prefill_during_decode", [(0.0, dict(length=512, nonce="pd-a", max_tokens=128)),
                                               (2.0, dict(length=8192, nonce="pd-b", max_tokens=16))])
        scenario("cancel_during_prefill", [(0.0, dict(length=8192, nonce="cx-a", max_tokens=16, cancel_after=1.5))])
        scenario("after_cancel_normal", [(0.0, dict(length=2048, nonce="cx-b", max_tokens=16))])
        scenario("stagger_short_ends_first", [(0.0, dict(length=8192, nonce="st-a", max_tokens=16)),
                                               (0.5, dict(length=256, nonce="st-b", max_tokens=8))])
        if args.extra:
            scenario("c4_four_4096_prefills", [(0.1 * i, dict(length=4096, nonce=f"c4-{i}", max_tokens=16)) for i in range(4)])
            scenario("c4_decode_plus_three_prefills", [(0.0, dict(length=512, nonce="c4d-a", max_tokens=128))] +
                     [(2.0 + 0.2 * i, dict(length=4096, nonce=f"c4d-{i}", max_tokens=16)) for i in range(3)])
            if args.cache:
                # a request that shares a cached prefix is cancelled during its prefill; the cache must stay usable
                request(4096, "cc-base", 8)
                scenario("cancel_during_cached_prefill", [(0.0, dict(length=8192, nonce="cc-base", max_tokens=16, cancel_after=1.0))])
                scenario("same_prompt_after_cancel", [(0.0, dict(length=4096, nonce="cc-base", max_tokens=16))])
            import urllib.request as ur
            def post(path):
                return ur.urlopen(ur.Request(url + path, data=b"", method="POST"), timeout=600).read()
            res["unload_reload"] = []
            for cyc in range(2):
                post(f"/v1/models/{args.model}/unload")
                time.sleep(4.0)
                after_unload = ctl.call("mem")
                post(f"/v1/models/{args.model}/load")
                time.sleep(4.0)
                for _ in range(60):
                    fl2 = ctl.call("flags")
                    if fl2["mlp_patched"] and fl2["gdn_registered"]:
                        break
                    time.sleep(0.5)
                assert ctl.call("install")["installed"]
                r2 = request(2048, f"reload-{cyc}", 16)
                time.sleep(12.0)
                res["unload_reload"].append({"cycle": cyc, "after_unload": after_unload, "request": r2,
                                             "after_request": sample(f"reload{cyc}")})
                print(f"[{args.arm}] reload {cyc}: ttft={r2.get('ttft') and round(r2['ttft'], 3)} "
                      f"unloaded_active={after_unload['active']/GiB:.2f} "
                      f"active={res['unload_reload'][-1]['after_request']['mem']['active']/GiB:.3f}", flush=True)
                save()
        loop = []
        for i in range(args.loop_requests):
            length = 512 if i % 2 == 0 else 2048
            r = request(length, f"loop-{i}", 32)
            time.sleep(6.0)
            sm = sample(f"loop{i}")
            loop.append({"i": i, "request": r, "sample": sm})
            res["loop"] = loop
            save()
            print(f"[{args.arm}] loop {i:>2} L={length} ttft={r.get('ttft') and round(r['ttft'], 3)} "
                  f"active={sm['mem']['active']/GiB:.3f} phys={sm['mem']['phys_footprint']/GiB:.3f} "
                  f"holders={(sm['lifetime'] or {}).get('holders')}", flush=True)
            if r.get("error"):
                raise SystemExit(f"stopping loop: {r['error']}")
        time.sleep(20.0)
        res["final_sample"] = sample("final")
    finally:
        m.stop_server(proc)
        txt = log_path.read_text(errors="replace") if log_path.exists() else ""
        res["log_suspect_lines"] = [l[:300] for l in txt.splitlines()
                                    if any(w in l for w in ("Traceback", "ERROR", "panic", "underflow", "timeout"))][:40]
        save()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
