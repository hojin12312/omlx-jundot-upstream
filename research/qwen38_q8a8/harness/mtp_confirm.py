#!/usr/bin/env python3
"""MTP + Q8A8 confirmation: prefill 4K with A8 on and MTP on, then decode 64 tokens (draft and verify steps run).

Reports: whether MTP engaged (server log, response usage), the A8 projection delta of the prefill-only request
(max_tokens=1) against the decode request (max_tokens=64) for the same prompt, and the generated text.
"""
import argparse, hashlib, json, os, shutil, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_pp2 as m  # noqa: E402
from measure_q8 import HOOK_DIR, Ctl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--python", required=True); ap.add_argument("--tree", required=True)
ap.add_argument("--workdir", required=True); ap.add_argument("--model-dir", required=True)
ap.add_argument("--checkpoint", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--corpus", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--mtp-key", default="vlm_mtp_enabled")
args = ap.parse_args()
for k in [k for k in os.environ if k.startswith("OMLX_") or k == "PYTHONPATH"]:
    del os.environ[k]
base = Path(args.workdir) / "mtp_confirm"
shutil.rmtree(base, ignore_errors=True); (base / "logs").mkdir(parents=True); (base / "home").mkdir()
settings = {"version": 1.0, "server": {"host": "127.0.0.1", "log_level": "info"},
            "model": {"model_dir": args.model_dir, "model_fallback": False}, "cache": {"enabled": False},
            "scheduler": {"max_concurrent_requests": 1, "chunked_prefill": True},
            "sampling": {"temperature": 0.0, "max_tokens": 4096},
            "auth": {"api_key": None, "skip_api_key_verification": True},
            "logging": {"log_dir": str(base / "logs"), "retention_days": 1},
            "memory": {"prefill_memory_guard": False}, "huggingface": {"hf_cache_enabled": False}}
m.write_settings(base, settings)
(base / "model_settings.json").write_text(json.dumps({"version": 1, "models": {args.model: {
    "qwen35_oq_a8_enabled": True, "qwen35_oq_a8_min_tokens": 128, args.mtp_key: True}}}, indent=2))
ctl = Ctl(base / "ctl")
env = {"PYTHONPATH": args.tree + ":" + str(HOOK_DIR), "HOME": str(base / "home"), "OMLX_Q8_CTL": str(base / "ctl")}
port = m.free_port(8350); url = f"http://127.0.0.1:{port}"
log = base / "logs" / "server.log"
proc = m.start_server(args.python, base, Path(args.model_dir), port, env, log, [])
tok = m.load_tokenizer(Path(args.checkpoint)); flat = json.load(open(args.corpus))["tokens"]
ids = tok.encode("[mtp confirm] ").ids; ids += flat[1000:1000 + 4096 - len(ids)]
prompt = tok.decode(ids)
res = {}
def req(max_tokens):
    body = json.dumps({"model": args.model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0, "stream": False}).encode()
    r = urllib.request.Request(url + "/v1/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time(); out = json.load(urllib.request.urlopen(r, timeout=900)); out["_s"] = time.time() - t0; return out
try:
    m.wait_ready(url + "/v1/models", proc)
    req(1)  # loads the model, installs the hook
    for _ in range(120):
        snap = ctl.call("snap")
        if snap["installed"] or "install_error" in snap["meta"]:
            break
        time.sleep(0.5)
    ctl.call("set", state="on")
    for name, mt in (("prefill_only", 1), ("decode64", 64)):
        b = ctl.call("snap"); out = req(mt); time.sleep(0.5); a = ctl.call("snap")
        res[name] = {"max_tokens": mt, "counts": {k: a["counts"].get(k, 0) - b["counts"].get(k, 0) for k in set(a["counts"]) | set(b["counts"])},
                     "usage": out["usage"], "text": out["choices"][0]["text"], "seconds": out["_s"]}
finally:
    m.stop_server(proc)
txt = log.read_text(errors="replace")
res["log_mtp_lines"] = [l for l in txt.splitlines() if "mtp" in l.lower() and ("disabled" in l.lower() or "enabled" in l.lower() or "engag" in l.lower() or "draft" in l.lower())][:15]
res["log_conflict_disabled"] = any("vlm_mtp_enabled disabled on load" in l for l in txt.splitlines())
Path(args.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)[:3000])
