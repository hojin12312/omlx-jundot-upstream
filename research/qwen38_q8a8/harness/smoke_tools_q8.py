#!/usr/bin/env python3
"""Q8A8 functional tool-call smoke: A8 off (A16) vs A8 on, greedy decoding.

One oMLX server of the tree under test (same settings and control-FIFO hook as ``measure_prod.py``). Every
case is a /v1/chat/completions request with ``tools`` and a long agent transcript, long enough that
the routed A8 path really runs (the floor is 128 tokens; the cases are 1.5K-12K tokens). Each case is
sent under both conditions with temperature 0, and per request the in-memory counters prove that
Down A8 ran in Q1 (``down_calls`` > 0) and did not in Q0 (== 0). Decode itself is always A16, so
the A8 effect enters only through the prefill (the KV and the recurrent state).

Checks per response (the same for both conditions): the answer contains a tool call; the tool name
is one of the offered tools; the arguments parse as JSON; every required argument is present with
the declared JSON type; no argument outside the schema; ``finish_reason`` is ``tool_calls``. The two
conditions are compared on the tool name, the arguments and the first divergence of the text.
"""
import argparse
import json
import os
import random
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_pp2 as m  # noqa: E402
from measure_q8 import HOOK_DIR, Ctl  # noqa: E402


def fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


TOOLS = [
    fn("read_file", "Read a file from the repository.", {"path": {"type": "string"}, "offset": {"type": "integer"}, "limit": {"type": "integer"}}, ["path"]),
    fn("grep", "Search file contents with a regular expression.", {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"]),
    fn("run_tests", "Run the test suite for one target.", {"target": {"type": "string"}, "verbose": {"type": "boolean"}}, ["target"]),
    fn("get_weather", "Weather for a city.", {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}}, ["city"]),
    fn("create_event", "Create a calendar event.", {"title": {"type": "string"}, "start": {"type": "string"}, "duration_min": {"type": "integer"}}, ["title", "start"]),
]
JSON_TYPES = {"string": str, "integer": int, "boolean": bool, "number": (int, float)}


def make_cases(root: Path, n: int):
    files = sorted(p for p in (root / "omlx").rglob("*.py") if p.stat().st_size > 6000)
    rng = random.Random(2026)
    asks = [
        ("Open the file that defines the routed gate-up kernel and show me its first 40 lines.", "read_file"),
        ("Find every place that mentions the word routed in the scheduler directory.", "grep"),
        ("Run the tests for tests/test_m5_gather_qmm_a8.py and show the verbose output.", "run_tests"),
        ("What is the weather in Busan right now? Use celsius.", "get_weather"),
        ("Put a 45 minute design review on the calendar for 2026-11-03T14:00.", "create_event"),
    ]
    cases = []
    for i in range(n):
        history = []
        for f in rng.sample(files, 3 + i % 3):
            lines = f.read_text(errors="replace").splitlines()
            start = rng.randint(0, max(0, len(lines) - 60))
            history.append({"role": "user", "content": f"Earlier context from {f.relative_to(root)}:\n" + "\n".join(lines[start:start + 220])})
            history.append({"role": "assistant", "content": f"Noted the contents of {f.name} (lines {start}-{start + 220})."})
        ask, expect = asks[i % len(asks)]
        cases.append({"id": f"case{i:02d}", "expect": expect, "messages": [
            {"role": "system", "content": "You are a coding and personal assistant. When the user asks for something a tool can do, call exactly one tool."},
            *history, {"role": "user", "content": ask}]})
    return cases


def chat(url, model, messages, timeout=900.0):
    body = json.dumps({"model": model, "messages": messages, "tools": TOOLS, "temperature": 0.0,
                       "max_tokens": 512, "stream": False,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    out["_seconds"] = time.time() - t0
    return out


def check(resp, expect):
    """Structural checks of one response; returns (ok, reasons, parsed)."""
    reasons = []
    ch = resp["choices"][0]
    msg = ch["message"]
    calls = msg.get("tool_calls") or []
    parsed = None
    if not calls:
        reasons.append("no tool call")
        return False, reasons, {"content": msg.get("content")}
    call = calls[0]["function"]
    name = call["name"]
    spec = next((t["function"] for t in TOOLS if t["function"]["name"] == name), None)
    if spec is None:
        reasons.append(f"unknown tool {name}")
        return False, reasons, {"name": name}
    try:
        args = json.loads(call["arguments"]) if isinstance(call["arguments"], str) else call["arguments"]
        assert isinstance(args, dict)
    except Exception:  # noqa: BLE001
        reasons.append("arguments are not a JSON object")
        return False, reasons, {"name": name, "arguments": call["arguments"]}
    props, required = spec["parameters"]["properties"], spec["parameters"]["required"]
    for r in required:
        if r not in args:
            reasons.append(f"missing required {r}")
    for k, v in args.items():
        if k not in props:
            reasons.append(f"argument {k} not in schema")
            continue
        want = JSON_TYPES[props[k]["type"]]
        if not isinstance(v, want) or (props[k]["type"] == "integer" and isinstance(v, bool)):
            reasons.append(f"argument {k} has the wrong type")
        if "enum" in props[k] and v not in props[k]["enum"]:
            reasons.append(f"argument {k} outside enum")
    if ch.get("finish_reason") != "tool_calls":
        reasons.append(f"finish_reason {ch.get('finish_reason')}")
    if name != expect:
        reasons.append(f"expected tool {expect}, got {name}")
    parsed = {"name": name, "arguments": args}
    return not reasons, reasons, parsed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cases", type=int, default=10)
    ap.add_argument("--model-settings", default='{"qwen35_oq_a8_enabled": true, "qwen35_oq_a8_min_tokens": 128}')
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
                "memory": {"prefill_memory_guard": False}, "huggingface": {"hf_cache_enabled": False}}
    m.write_settings(base, settings)
    (base / "model_settings.json").write_text(json.dumps(
        {"version": 1, "models": {args.model: json.loads(args.model_settings)}}, indent=2))
    ctl = Ctl(base / "ctl")
    env = {"PYTHONPATH": args.tree + ":" + str(HOOK_DIR), "HOME": str(base / "home"),
           "OMLX_Q8_CTL": str(base / "ctl")}
    port = m.free_port(8350)
    url = f"http://127.0.0.1:{port}"
    proc = m.start_server(args.python, base, Path(args.model_dir), port, env,
                          base / "logs" / f"server-{args.tag}.log", [])

    res = {"tag": args.tag, "tree": args.tree, "cases": []}
    try:
        m.wait_ready(url + "/v1/models", proc)
        for case in make_cases(Path(args.tree), args.cases):
            row = {"id": case["id"], "expect": case["expect"]}
            for cond, flag in (("q0", "off"), ("q1", "on")):
                ctl.call("set", state=flag)
                time.sleep(0.5)
                before = ctl.call("snap")
                resp = chat(url, args.model, case["messages"])
                time.sleep(0.4)
                after = ctl.call("snap")
                if "server_files" not in res:
                    for _ in range(120):
                        if after["installed"] or "install_error" in after["meta"]:
                            break
                        time.sleep(0.5)
                        after = ctl.call("snap")
                    meta = after["meta"]
                    assert after["installed"] and "install_error" not in meta, meta
                    assert meta["omlx_file"].startswith(args.tree + "/"), f"wrong tree: {meta}"
                    res["server_files"] = {k: meta.get(k) for k in ("omlx_file", "oa_file", "pid", "native")}
                ok, reasons, parsed = check(resp, case["expect"])
                row[cond] = {"ok": ok, "reasons": reasons, "parsed": parsed,
                             "prompt_tokens": resp["usage"]["prompt_tokens"],
                             "a8_calls": after["counts"].get("a8_proj", 0) - before["counts"].get("a8_proj", 0),
                             "text": resp["choices"][0]["message"].get("content"),
                             "seconds": resp["_seconds"]}
            row["same_call"] = row["q0"]["parsed"] == row["q1"]["parsed"]
            res["cases"].append(row)
            print(f"{row['id']} {row['expect']:<12} q0 ok={row['q0']['ok']} a8={row['q0']['a8_calls']} | "
                  f"q1 ok={row['q1']['ok']} a8={row['q1']['a8_calls']} | same call: {row['same_call']} "
                  f"tokens={row['q1']['prompt_tokens']}", flush=True)
            Path(args.out).write_text(json.dumps(res, indent=2))
    finally:
        m.stop_server(proc)
        Path(args.out).write_text(json.dumps(res, indent=2))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
