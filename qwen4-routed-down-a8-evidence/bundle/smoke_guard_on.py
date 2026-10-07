#!/usr/bin/env python3
"""Normal-guard 8K serving smoke: the oMLX server with ``prefill_memory_guard`` ON (the product default).

The timing campaign runs with the scheduler prefill memory guard off in both conditions. This smoke shows the
same candidate serving real 8192-token requests with the guard on: every request must finish with one token and
the Down A8 path must run (in-memory counters through the hook of ``prod_toggle``). No condition switching.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_pp2 as m  # noqa: E402
from measure_prod import Ctl, HOOK_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--python", required=True)
ap.add_argument("--tree", required=True)
ap.add_argument("--workdir", required=True)
ap.add_argument("--model-dir", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--corpus", required=True)
ap.add_argument("--length", type=int, default=8192)
ap.add_argument("--requests", type=int, default=3)
ap.add_argument("--settle", type=float, default=0.4, help="seconds between requests (the guard counts memory that the previous request has not released yet)")
ap.add_argument("--tag", required=True)
ap.add_argument("--out", "-o", required=True)
args = ap.parse_args()

for k in [k for k in os.environ if k.startswith("OMLX_") or k == "PYTHONPATH"]:
    del os.environ[k]
base = Path(args.workdir) / f"smoke_{args.tag}"
shutil.rmtree(base, ignore_errors=True)
(base / "logs").mkdir(parents=True)
(base / "home").mkdir()
settings = {"version": 1.0, "server": {"host": "127.0.0.1", "log_level": "info"},
            "model": {"model_dir": args.model_dir, "model_fallback": False}, "cache": {"enabled": False},
            "scheduler": {"max_concurrent_requests": 1, "chunked_prefill": True},
            "sampling": {"temperature": 0.0, "max_tokens": 4096},
            "auth": {"api_key": None, "skip_api_key_verification": True},
            "logging": {"log_dir": str(base / "logs"), "retention_days": 1},
            "memory": {"prefill_memory_guard": True}, "huggingface": {"hf_cache_enabled": False}}
m.write_settings(base, settings)
(base / "model_settings.json").write_text(json.dumps(
    {"version": 1, "models": {args.model: {"qwen35_oq_a8_enabled": True, "qwen35_oq_a8_min_tokens": 128}}}, indent=2))
ctl = Ctl(base / "ctl")
env = {"PYTHONPATH": args.tree + ":" + str(HOOK_DIR), "HOME": str(base / "home"), "OMLX_PROD_CTL": str(base / "ctl")}
tok = m.load_tokenizer(Path(args.checkpoint))
doc = json.load(open(args.corpus))
flat = doc["tokens"]
port = m.free_port(8350)
url = f"http://127.0.0.1:{port}"
proc = m.start_server(args.python, base, Path(args.model_dir), port, env, base / "logs" / f"server-{args.tag}.log", [])
res = {"tag": args.tag, "tree": args.tree, "prefill_memory_guard": True, "settings": settings, "requests": []}
try:
    m.wait_ready(url + "/v1/models", proc)
    for i in range(args.requests + 1):
        nonce = f"smoke-{args.length}-{i}"
        salt = tok.encode(f"[sample {nonce}] ").ids
        off = int(hashlib.sha256(nonce.encode()).hexdigest(), 16) % len(flat)
        ids = salt + (flat[off:] + flat[:off])[: args.length - len(salt)]
        body = m.request_body(args.model, tok.decode(ids), 1)
        time.sleep(args.settle)
        before = ctl.call("snap")
        rec = m.timed_prefill_strict(url, body, 1800.0)
        time.sleep(0.4)
        after = ctl.call("snap")
        down = after["counts"].get("down_a8_results", 0) - before["counts"].get("down_a8_results", 0)
        gate = after["counts"].get("gate_up_a8_results", 0) - before["counts"].get("gate_up_a8_results", 0)
        rows = max((int(k) for k in after["down_rows"]), default=0)
        rec.update({"phase": "warmup" if i == 0 else "timed", "down_a8_results": down, "gate_up_a8_results": gate,
                    "max_down_rows_per_call": rows, "request_body_sha256": hashlib.sha256(body).hexdigest()})
        res["requests"].append(rec)
        print(f"[{args.tag}] request {i} {rec['phase']} pt={rec['prompt_tokens']} token_seen={rec['token_seen']} "
              f"ttft={rec['ttft_s']} down_a8={down} gate_up_a8={gate}", flush=True)
        if i == 0:
            meta = after["meta"]
            assert after["installed"] and meta["omlx_file"].startswith(args.tree + "/"), meta
            res["server_files"] = {k: meta.get(k) for k in ("omlx_file", "a8_file", "pid", "native")}
    timed = [r for r in res["requests"] if r["phase"] == "timed"]
    res["pass"] = all(r["token_seen"] and r["completion_tokens"] == 1 and r["down_a8_results"] > 0 for r in timed)
    print("PASS" if res["pass"] else "FAIL", flush=True)
finally:
    m.stop_server(proc)
    Path(args.out).write_text(json.dumps(res, indent=2))
