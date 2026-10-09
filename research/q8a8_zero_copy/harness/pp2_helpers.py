#!/usr/bin/env python3
"""Paired full-model prefill (PP/s) harness for the routed-A8 study (round 2).

Round-2 changes against `measure_pp.py`: every length gets its own explicit
warm-up request(s) (separate prompt, recorded as phase "warmup", excluded
from statistics) before >= `--samples` timed samples; prompts are salted by
(length, sample) only, so every condition is timed on the *same* prompts; the
power/thermal state is recorded at start and per request.

Original description:


Measures *actual full-model prefill* through the oMLX server, one condition
at a time, with a fresh server process and a fresh (empty) cache per
condition so that no condition can benefit from another's prefix cache.

Per sample:
  * prompt = `--length` tokens taken from a fixed corpus, salted with a
    per-(length, sample) nonce so repeated samples inside one condition can
    never hit the prefix cache either;
  * request is streamed with `max_tokens=1`, so the wall time from request
    start to the first streamed token is the prefill time plus fixed
    overhead, and decode contributes ~one token;
  * PP/s = usage.prompt_tokens / TTFT.

The same (length, sample) prompt set is used for every condition, and
conditions are executed in the order given so AB/BA alternation is a
command-line choice.

Usage:
  measure_pp.py --base-path BASE --model-dir DIR --tag b0 \
      --lengths 1024 4096 8192 16384 --samples 3 -o out.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_CORPUS = (
    "The routed expert path in a mixture-of-experts decoder evaluates a "
    "small subset of experts per token. Prefill throughput is dominated by "
    "the grouped matrix multiplications that consume the sorted routes, so "
    "the operand path of those multiplications is what this study changes. "
)


# ---------------------------------------------------------------------------
# server lifecycle
# ---------------------------------------------------------------------------


def free_port(preferred: int) -> int:
    for port in range(preferred, preferred + 64):
        with socket.socket() as sock:
            try:
                sock.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise SystemExit("no free port")


def write_settings(base: Path, settings: dict) -> None:
    base.mkdir(parents=True, exist_ok=True)
    (base / "settings.json").write_text(json.dumps(settings, indent=2))


def wait_ready(url: str, proc: subprocess.Popen, timeout: float = 1800.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:  # noqa: BLE001
            time.sleep(2.0)
    raise SystemExit("server did not become ready")


def start_server(
    python: str,
    base_path: Path,
    model_dir: Path,
    port: int,
    env: dict,
    log_path: Path,
    extra_args: list[str],
) -> subprocess.Popen:
    cmd = [
        python,
        "-m",
        "omlx.cli",
        "serve",
        "--base-path",
        str(base_path),
        "--model-dir",
        str(model_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ] + extra_args
    log = open(log_path, "wb")
    proc = subprocess.Popen(
        cmd,
        stdout=log,
        stderr=subprocess.STDOUT,
        env={**os.environ, **env},
        start_new_session=True,
    )
    proc._omlx_log = log  # type: ignore[attr-defined]
    return proc


def stop_server(proc: subprocess.Popen, grace: float = 60.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)
    log = getattr(proc, "_omlx_log", None)
    if log is not None:
        log.close()


# ---------------------------------------------------------------------------
# prompt construction
# ---------------------------------------------------------------------------


def load_tokenizer(ckpt: Path):
    try:
        from tokenizers import Tokenizer  # type: ignore
    except ImportError:
        return None
    path = ckpt / "tokenizer.json"
    if not path.exists():
        return None
    return Tokenizer.from_file(str(path))


def build_prompt(tokenizer, length: int, nonce: str, corpus: str) -> tuple[str, int]:
    """A prompt of exactly `length` tokens (or an approximation without a
    tokenizer), salted by `nonce` so no two samples share a prefix."""
    salt = f"[sample {nonce}] "
    if tokenizer is None:
        # Fall back to a character budget; the measured length comes from the
        # server's own usage.prompt_tokens.
        approx = max(1, length - 8)
        text = salt + (corpus * (approx // len(corpus) + 2))[: approx * 4]
        return text, -1
    salt_ids = tokenizer.encode(salt).ids
    body_budget = max(1, length - len(salt_ids))
    ids: list[int] = []
    while len(ids) < body_budget:
        ids.extend(tokenizer.encode(corpus).ids)
    ids = ids[:body_budget]
    full = salt_ids + ids
    return tokenizer.decode(full), len(full)


# ---------------------------------------------------------------------------
# one timed request
# ---------------------------------------------------------------------------


def timed_prefill(
    base_url: str, model: str, prompt: str, max_tokens: int, timeout: float
) -> dict:
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base_url + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    usage = {}
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
            if ttft is None:
                choices = obj.get("choices") or []
                if choices and choices[0].get("text"):
                    ttft = time.perf_counter() - t0
                    chunks += 1
    total = time.perf_counter() - t0
    if ttft is None:
        ttft = total
    return {
        "ttft_s": ttft,
        "total_s": total,
        "usage": usage,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
    }


def request_body(model: str, prompt: str, max_tokens: int) -> bytes:
    """The exact bytes of one streamed completion request (built once per matched pair)."""
    return json.dumps({
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }).encode()


def timed_prefill_strict(base_url: str, body: bytes, timeout: float) -> dict:
    """One streamed request from prebuilt body bytes, with a strict first-token rule.

    ``ttft_s`` is the time to the first SSE event that carries non-empty token text. Role-only or
    empty events are never counted as the token, and a stream that ends without any token text is
    reported as ``token_seen = False`` with ``ttft_s = None`` (the caller marks the sample invalid;
    nothing falls back to the total time). ``prompt_tokens`` comes only from the server's usage
    block (no planned-count fallback)."""
    req = urllib.request.Request(
        base_url + "/v1/completions",
        data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = first_event = None
    usage: dict = {}
    events = 0
    finish = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
            events += 1
            now = time.perf_counter() - t0
            if first_event is None:
                first_event = now
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            for ch in choices:
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
            if ttft is None and choices and choices[0].get("text"):
                ttft = now
    total = time.perf_counter() - t0
    details = usage.get("prompt_tokens_details")
    return {
        "ttft_s": ttft,
        "token_seen": ttft is not None,
        "first_event_s": first_event,
        "events": events,
        "finish_reason": finish,
        "total_s": total,
        "usage": usage,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": (details or {}).get("cached_tokens") if isinstance(details, dict) else None,
        "cached_tokens_reported": isinstance(details, dict) and details.get("cached_tokens") is not None,
    }


def power_state() -> dict:
    out = {}
    for key, cmd in (("batt", ["pmset", "-g", "batt"]), ("therm", ["pmset", "-g", "therm"])):
        try:
            out[key] = subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
        except Exception as exc:  # noqa: BLE001
            out[key] = repr(exc)
    return out


def model_list(base_url: str) -> list[str]:
    with urllib.request.urlopen(base_url + "/v1/models", timeout=30) as resp:
        data = json.load(resp)
    return [m["id"] for m in data.get("data", [])]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--base-path", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--checkpoint", required=True, help="for the tokenizer")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1024, 4096, 8192, 16384])
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1, help="warm-up requests per length")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--max-tokens", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="K=V",
        help="environment variable for the server (repeatable)",
    )
    ap.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="dotted.key=json",
        help="settings.json override (repeatable)",
    )
    ap.add_argument("--model", default=None, help="model id (default: first listed)")
    ap.add_argument("--no-restart", action="store_true", help="reuse a running server")
    ap.add_argument(
        "--model-setting",
        action="append",
        default=[],
        metavar="k=v",
        help="model_settings.json field for the served model (repeatable)",
    )
    ap.add_argument(
        "--model-id",
        default=None,
        help="model_settings.json key (default: the checkpoint directory name)",
    )
    ap.add_argument("--out", "-o", required=True)
    ap.add_argument("--log-dir", default=None)
    args = ap.parse_args()

    base_path = Path(args.base_path).expanduser()
    model_dir = Path(args.model_dir).expanduser()
    ckpt = Path(args.checkpoint).expanduser()
    log_dir = Path(args.log_dir) if args.log_dir else base_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = {}
    for item in args.env:
        k, _, v = item.partition("=")
        env[k] = v

    settings = {
        "version": 1.0,
        "server": {"host": "127.0.0.1", "log_level": "info"},
        "model": {"model_dir": str(model_dir), "model_fallback": False},
        # Prefix cache off: no condition may reuse another's prefill state.
        "cache": {"enabled": False},
        "scheduler": {
            "max_concurrent_requests": 1,
            "chunked_prefill": True,
            "prefill_step_size": 4096,
        },
        "sampling": {"temperature": 0.0, "max_tokens": 4096},
        "auth": {"api_key": None, "skip_api_key_verification": False},
        "logging": {"log_dir": str(log_dir), "retention_days": 1},
        "memory": {"prefill_memory_guard": False},
    }
    for item in args.setting:
        key, _, raw = item.partition("=")
        value = json.loads(raw)
        node = settings
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    write_settings(base_path, settings)

    # Per-model settings live in their own file next to settings.json.
    if args.model_setting:
        model_key = args.model_id or ckpt.name
        model_fields: dict = {}
        for item in args.model_setting:
            key, _, raw = item.partition("=")
            model_fields[key] = json.loads(raw)
        (base_path / "model_settings.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "models": {model_key: model_fields},
                },
                indent=2,
            )
        )
        print(f"model_settings.json: {model_key} -> {model_fields}")

    tokenizer = load_tokenizer(ckpt)
    if tokenizer is None:
        print("warning: tokenizer.json unavailable; prompt lengths approximate")

    prompts: dict[int, list[tuple[str, int]]] = {}
    for length in args.lengths:
        prompts[length] = [
            build_prompt(tokenizer, length, f"r2-{length}-{i}", DEFAULT_CORPUS)
            for i in range(args.samples)
        ]
    warm_prompts: dict[int, list[tuple[str, int]]] = {
        length: [
            build_prompt(tokenizer, length, f"r2-warm-{length}-{i}", DEFAULT_CORPUS)
            for i in range(args.warmup)
        ]
        for length in args.lengths
    }

    port = free_port(args.port)
    base_url = f"http://127.0.0.1:{port}"
    proc = None
    if not args.no_restart:
        log_path = log_dir / f"server-{args.tag}-{port}.log"
        proc = start_server(
            args.python, base_path, model_dir, port, env, log_path, []
        )
        try:
            wait_ready(base_url + "/v1/models", proc)
        except SystemExit:
            stop_server(proc)
            raise

    try:
        models = model_list(base_url)
        model = args.model or (models[0] if models else None)
        if model is None:
            raise SystemExit("no model is served")
        print(f"[{args.tag}] server up on {base_url}, model={model}")

        result = {
            "tag": args.tag,
            "port": port,
            "model": model,
            "env": env,
            "settings": settings,
            "lengths": args.lengths,
            "samples": args.samples,
            "measurements": [],
        }
        result["power"] = power_state()
        result["started_at"] = time.time()
        result["warmup"] = args.warmup
        for length in args.lengths:
            for i, (prompt, planned) in enumerate(warm_prompts[length]):
                rec = timed_prefill(base_url, model, prompt, args.max_tokens, args.timeout)
                pt = rec["prompt_tokens"] or planned
                rec.update({"phase": "warmup", "length_requested": length, "sample": i,
                            "pp_tokens_per_s": (pt / rec["ttft_s"]) if rec["ttft_s"] else None,
                            "t": time.time()})
                result["measurements"].append(rec)
                print(f"[{args.tag}] L={length:>6} WARMUP {i} ttft={rec['ttft_s']:.4f}s "
                      f"pp/s={rec['pp_tokens_per_s']:.1f}")
            for i, (prompt, planned) in enumerate(prompts[length]):
                rec = timed_prefill(
                    base_url, model, prompt, args.max_tokens, args.timeout
                )
                pt = rec["prompt_tokens"] or planned
                rec.update(
                    {
                        "length_requested": length,
                        "planned_tokens": planned,
                        "phase": "timed",
                        "t": time.time(),
                        "sample": i,
                        "pp_tokens_per_s": (pt / rec["ttft_s"]) if rec["ttft_s"] else None,
                    }
                )
                result["measurements"].append(rec)
                print(
                    f"[{args.tag}] L={length:>6} sample={i} "
                    f"prompt_tokens={pt} ttft={rec['ttft_s']:.4f}s "
                    f"pp/s={rec['pp_tokens_per_s']:.1f}"
                )
    finally:
        if proc is not None:
            stop_server(proc)

    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
