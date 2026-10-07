#!/usr/bin/env python3
"""Matched-input prefill throughput driver for the production branch (final version).

One oMLX server of the tree under test, production scheduler settings (prefix cache off, one request
at a time, chunked prefill), dense A8 and routed Gate+Up A8 on. Modes:

  paired (default)  one server, the routed Down projection is switched between requests (b0 = Down A16,
                    b1 = Down A8) through the control FIFO of ``prod_toggle/sitecustomize.py``; the
                    two requests of a pair carry the *same* body bytes. ``--first`` fixes which
                    condition runs first in the first pair; the order alternates AB/BA afterwards.
  aa                same as paired, but both labels run Down A16 (a null experiment of the design).
  single            one fixed condition per server (``--label``, optionally ``--down-env 0|1`` that sets
                    ``OMLX_M5_ROUTED_DOWN_A8`` for the process). Prompts depend only on
                    ``--nonce-set``, length and sample index, so two single runs with the same
                    nonce-set receive identical inputs. ``--hook off`` runs without any hook or
                    counter (the instrumentation-free control).

Matching: the prompt of a pair is built once, so both conditions receive identical request bytes.
sha256 of the body and of the re-tokenised prompt ids are stored per request and compared per pair;
condition labels are driver metadata and never enter the request. ``usage.prompt_tokens`` must exist
(no planned-count fallback) and be equal inside a pair. TTFT is the first SSE event with non-empty
token text; a stream without a token is invalid. An invalid pair is retried with a new nonce
(``--retries``), and invalid requests stay in the file, flagged, and are excluded by the analysis.

Hot path: the condition is switched and acknowledged (generation number) before a request starts and
the counters are read before and after it; nothing is read or written while a request is timed.

PP/s = ``usage.prompt_tokens`` / client-observed TTFT (max_tokens = 1). ``--prompt-source corpus``
cuts the salted prompt from a token corpus (``--corpus``) at a hashed offset.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_pp2 as m  # noqa: E402

HOOK_DIR = Path(__file__).resolve().parent / "prod_toggle"
FINGERPRINT_FILES = ("omlx/patches/m5_gather_qmm_a8.py", "omlx/patches/qwen35_moe_gate_up.py",
                     "omlx/patches/qwen35_moe_weighted_sum.py", "omlx/custom_kernels/nax.py")


def sh(*cmd):
    try:
        return subprocess.run(list(cmd), capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return repr(exc)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


class Ctl:
    """Driver side of the control FIFO: one command, then wait for the matching acknowledgement."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.fifo = directory / "ctl.fifo"
        self.ack = directory / "ack.json"
        self.gen = 0
        directory.mkdir(parents=True, exist_ok=True)
        os.mkfifo(self.fifo)

    def call(self, op, timeout=60.0, **kw):
        self.gen += 1
        line = (json.dumps({"op": op, "gen": self.gen, **kw}) + "\n").encode()
        deadline = time.time() + timeout
        while True:
            try:
                fd = os.open(self.fifo, os.O_WRONLY | os.O_NONBLOCK)  # ENXIO until the hook listens
                break
            except OSError:
                if time.time() > deadline:
                    raise SystemExit("control FIFO has no reader: the hook is not loaded")
                time.sleep(0.05)
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
        while time.time() < deadline:
            try:
                ack = json.loads(self.ack.read_text())
                if ack.get("gen") == self.gen:
                    return ack
            except (OSError, ValueError):
                pass
            time.sleep(0.005)
        raise SystemExit(f"no acknowledgement for generation {self.gen} ({op})")


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", type=int, nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--first", default="b0", choices=["b0", "b1"], help="condition that runs first in the first pair")
    ap.add_argument("--mode", default="paired", choices=["paired", "aa", "single"])
    ap.add_argument("--label", default="single", help="condition label of a single run")
    ap.add_argument("--down-env", default=None, choices=["0", "1"], help="OMLX_M5_ROUTED_DOWN_A8 of a single run")
    ap.add_argument("--hook", default="on", choices=["on", "off"], help="off: no hook, no counters (single mode)")
    ap.add_argument("--nonce-set", default=None, help="prompt family (default: --tag)")
    ap.add_argument("--prompt-source", default="corpus", choices=["corpus"])
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--model-settings", default='{"qwen35_oq_a8_enabled": true, "qwen35_oq_a8_min_tokens": 128}')
    ap.add_argument("--settle", type=float, default=0.5)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--resume", action="store_true", help="continue a previous --out of the same tag")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", "-o", required=True)
    args = ap.parse_args()
    if args.mode == "single" and args.hook == "off" and args.down_env is None:
        ap.error("--hook off needs --down-env")
    if args.mode != "single" and args.hook == "off":
        ap.error("--hook off is for single runs")
    nonce_set = args.nonce_set or args.tag
    out_path = Path(args.out)

    if args.mode == "single":
        conds = {args.label: None}
    else:
        conds = {"b0": "0", "b1": "0" if args.mode == "aa" else "1"}
    names = list(conds)
    if args.mode != "single" and args.first == "b1":
        names = names[::-1]

    # Inherited research switches must not leak into the server: only what is set below is used.
    for k in [k for k in os.environ if k.startswith("OMLX_") or k == "PYTHONPATH"]:
        del os.environ[k]
    base = Path(args.workdir) / f"prod_{args.tag}"
    if not args.resume:
        shutil.rmtree(base, ignore_errors=True)
    (base / "logs").mkdir(parents=True, exist_ok=True)
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
    # The server writes a few files below $HOME (for instance ~/.omlx/bin/omlx-cluster-python); give it a
    # private HOME so that a measurement never touches the user's own oMLX directory.
    (base / "home").mkdir(exist_ok=True)
    env = {"PYTHONPATH": args.tree, "HOME": str(base / "home")}
    ctl = None
    if args.hook == "on":
        env["PYTHONPATH"] += ":" + str(HOOK_DIR)
        ctl_dir = base / "ctl"
        shutil.rmtree(ctl_dir, ignore_errors=True)
        ctl = Ctl(ctl_dir)
        env["OMLX_PROD_CTL"] = str(ctl_dir)
        if args.mode != "single":
            env["OMLX_PROD_TOGGLE"] = "1"
    if args.down_env is not None:
        env["OMLX_M5_ROUTED_DOWN_A8"] = args.down_env

    # Evidence of what the server will import and with which toolchain, from the same environment.
    probe_env = {**os.environ, **env}
    probe = json.loads(subprocess.run(
        [args.python, "-c",
         "import json,sys,importlib.metadata as md,omlx;"
         "print(json.dumps({'omlx_file':omlx.__file__,'python':sys.version.split()[0],"
         "'mlx':md.version('mlx'),'mlx_lm':md.version('mlx-lm'),'tokenizers':md.version('tokenizers')}))"],
        capture_output=True, text=True, env=probe_env, check=True).stdout.strip().splitlines()[-1])
    assert probe["omlx_file"].startswith(args.tree + "/"), f"wrong tree: {probe}"
    tree_git = sh("git", "-C", args.tree, "rev-parse", "HEAD")
    gpu_cores = re.search(r"Total Number of Cores:\s*(\d+)", sh("system_profiler", "SPDisplaysDataType"))
    host = {"hw_model": sh("sysctl", "-n", "hw.model"), "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"),
            "memory_gib": int(sh("sysctl", "-n", "hw.memsize")) / 2 ** 30, "macos": sh("sw_vers", "-productVersion"),
            "gpu_cores": int(gpu_cores.group(1)) if gpu_cores else None}
    fingerprints = {f: sha256_file(Path(args.tree) / f) for f in FINGERPRINT_FILES if (Path(args.tree) / f).exists()}

    tok = m.load_tokenizer(Path(args.checkpoint))
    assert tok is not None, "tokenizer.json is required"
    corpus_doc = json.load(open(args.corpus))
    flat = corpus_doc["tokens"] if "tokens" in corpus_doc else [t for s in corpus_doc["sequences"] for t in s["tokens"]]
    corpus_sha = hashlib.sha256(json.dumps(flat, separators=(",", ":")).encode()).hexdigest()

    def build(length, nonce):
        salt_ids = tok.encode(f"[sample {nonce}] ").ids
        n = length - len(salt_ids)
        assert 0 < n <= len(flat), "corpus shorter than the requested prompt"
        off = int(hashlib.sha256(nonce.encode()).hexdigest(), 16) % len(flat)
        ids = salt_ids + (flat[off:] + flat[:off])[:n]
        return tok.decode(ids), ids

    port = m.free_port(8350)
    url = f"http://127.0.0.1:{port}"
    proc = m.start_server(args.python, base, Path(args.model_dir), port, env,
                          base / "logs" / f"server-{args.tag}.log", [])
    res = {"schema": 2, "tag": args.tag, "mode": args.mode, "lengths": args.lengths, "samples": args.samples,
           "first": args.first, "nonce_set": nonce_set, "label": args.label, "hook": args.hook,
           "design": "matched prompt per pair, per-pair condition order AB/BA, salted prompts, prefix cache off",
           "argv": sys.argv, "tree": args.tree, "tree_git_head": tree_git, "env": env, "settings": settings,
           "model_settings": json.loads(args.model_settings), "host": host, "versions": probe,
           "fingerprints": fingerprints, "corpus": {"path": str(args.corpus), "tokens": len(flat),
                                                    "sha256": corpus_sha},
           "prefill_memory_guard": settings["memory"]["prefill_memory_guard"], "measurements": []}
    if args.resume and out_path.exists():
        prev = json.load(open(out_path))
        assert prev.get("tag") == args.tag and prev.get("schema") == 2, "cannot resume this file"
        res["measurements"] = prev["measurements"]
        res["resumed"] = res.get("resumed", 0) + 1

    def save():
        out_path.write_text(json.dumps(res, indent=2))

    def done_pairs():
        ok = {}
        for r in res["measurements"]:
            if r["phase"] == "timed" and r["valid"]:
                ok.setdefault((r["length_requested"], r["sample"]), set()).add(r["cond"])
        return {k for k, v in ok.items() if v == set(names)}

    def send(length, nonce, prompt, body, ids_info, cond, phase, i, attempt):
        down = conds[cond]
        before = after = ack = None
        if ctl is not None:
            if down is not None:
                ack = ctl.call("set", down=down)
            time.sleep(args.settle)
            before = ctl.call("snap")
        else:
            time.sleep(args.settle)
        pw = m.power_state()
        rec = m.timed_prefill_strict(url, body, args.timeout)
        time.sleep(0.4)
        if ctl is not None:
            after = ctl.call("snap")
        pt = rec["prompt_tokens"]
        rec.update({"cond": cond, "phase": phase, "sample": i, "attempt": attempt, "length_requested": length,
                    "nonce": nonce, "t": time.time(), "set_gen": ack["gen"] if ack else None,
                    "down_state_ack": ack["down"] if ack else None,
                    "request_body_sha256": hashlib.sha256(body).hexdigest(), **ids_info,
                    "pp_tokens_per_s": (pt / rec["ttft_s"]) if (pt and rec["ttft_s"]) else None,
                    "power_before": pw["batt"].splitlines()[0] if pw["batt"] else "",
                    "therm_before": pw["therm"].strip().replace("\n", " | ")})
        if before is not None:
            diff = lambda key: {k: after[key].get(k, 0) - before[key].get(k, 0)  # noqa: E731
                                for k in set(after[key]) | set(before[key])
                                if after[key].get(k, 0) != before[key].get(k, 0)}
            rec["coverage"] = {"counts": diff("counts"), "launch": diff("launch"), "mode_query": diff("mode_query"),
                               "down_rows": diff("down_rows")}
            if "server_files" not in res:
                meta = after["meta"]
                assert after["installed"] and "install_error" not in meta, meta
                assert meta["omlx_file"].startswith(args.tree + "/") and meta["a8_file"].startswith(args.tree + "/"), meta
                res["server_files"] = {k: meta.get(k) for k in ("omlx_file", "a8_file", "pid", "native", "has_down_code")}
        res["measurements"].append(rec)
        return rec

    def problems(rec):
        out = []
        if not rec["token_seen"]:
            out.append("no token text in the stream")
        if not rec["prompt_tokens"]:
            out.append("usage.prompt_tokens missing")
        if rec["cached_tokens_reported"] and rec["cached_tokens"] != 0:
            out.append(f"cached_tokens={rec['cached_tokens']}")
        if rec["completion_tokens"] != 1:
            out.append(f"completion_tokens={rec['completion_tokens']}")
        cov = rec.get("coverage")
        if cov is not None and conds.get(rec["cond"]) is not None:
            down = cov["counts"].get("down_a8_results", 0)
            want_down = conds[rec["cond"]] == "1"
            if want_down and down == 0:
                out.append("Down A8 did not run in an A8 condition")
            if not want_down and down != 0:
                out.append(f"Down A8 ran {down}x in an A16 condition")
        return out

    try:
        m.wait_ready(url + "/v1/models", proc)
        res["power"] = m.power_state()
        res["started_at"] = time.time()
        for length in args.lengths:
            todo = [i for i in range(args.samples) if (length, i) not in done_pairs()]
            if not todo:
                continue
            warm_prompt, warm_ids = build(length, f"prod-warm-{nonce_set}-{length}")
            warm_body = m.request_body(args.model, warm_prompt, 1)
            for cond in names:
                r = send(length, f"warm-{length}", warm_prompt, warm_body, {}, cond, "warmup", 0, 0)
                r["valid"] = True
                print(f"[{args.tag}] L={length:>6} warmup {cond:<3} ttft={r['ttft_s']} pt={r['prompt_tokens']}", flush=True)
            for i in todo:
                for attempt in range(1 + args.retries):
                    nonce = f"prod-{nonce_set}-{length}-{i}" + (f"-r{attempt}" if attempt else "")
                    prompt, ids = build(length, nonce)
                    body = m.request_body(args.model, prompt, 1)
                    re_ids = tok.encode(prompt).ids
                    ids_info = {"planned_ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                                "prompt_ids_sha256": hashlib.sha256(json.dumps(re_ids).encode()).hexdigest(),
                                "prompt_ids_count": len(re_ids)}
                    order = names if i % 2 == 0 else names[::-1]  # names is already rotated by --first
                    pair = []
                    for cond in order:
                        r = send(length, nonce, prompt, body, ids_info, cond, "timed", i, attempt)
                        pair.append(r)
                    bad = [p for r in pair for p in problems(r)]
                    if len({r["request_body_sha256"] for r in pair}) != 1 or len({r["prompt_ids_sha256"] for r in pair}) != 1:
                        raise SystemExit("matched pair carries different request bytes: harness bug")
                    if len({r["prompt_tokens"] for r in pair}) != 1:
                        bad.append(f"prompt_tokens differ inside the pair: {[r['prompt_tokens'] for r in pair]}")
                    for r in pair:
                        r["valid"] = not bad
                        r["invalid_reasons"] = bad
                    print(f"[{args.tag}] L={length:>6} timed {i} try{attempt} " + " | ".join(
                        f"{r['cond']} pt={r['prompt_tokens']} ttft={r['ttft_s'] and round(r['ttft_s'], 3)} "
                        f"pp/s={r['pp_tokens_per_s'] and round(r['pp_tokens_per_s'], 1)} "
                        f"down={r.get('coverage', {}).get('counts', {}).get('down_a8_results')}" for r in pair)
                          + (f" INVALID {bad}" if bad else ""), flush=True)
                    save()
                    if not bad:
                        break
        res["power_end"] = m.power_state()
    finally:
        m.stop_server(proc)
        save()
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
