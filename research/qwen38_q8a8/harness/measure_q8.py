#!/usr/bin/env python3
"""Matched-input prefill throughput driver for the Q8A8 study.

One oMLX server of the tree under test (production scheduler settings, prefix cache off, one request
at a time, chunked prefill, MTP off). The condition is switched between requests through the control
FIFO of ``q8_toggle/sitecustomize.py``: ``off`` (oQ A8 wrappers removed, the production baseline),
``on`` (Q8A8 wiring) and optionally ``a16`` (existing native Q8 A16 tile forced to the Q4/Q5 floor).
The requests of one tuple carry the same body bytes, and the order rotates per sample (AB/BA for
two conditions). PP/s = usage.prompt_tokens / client-observed TTFT (max_tokens = 1).

Counters (in-memory, identical wrappers in every state) are snapshotted before and after each request.
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

HOOK_DIR = Path(__file__).resolve().parent / "q8_toggle"
FINGERPRINT_FILES = ("omlx/patches/qwen35_oq_a8.py", "omlx/patches/qwen35_q4_mlp.py",
                     "omlx/custom_kernels/qwen35_prefill/fast.py")


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
    ap.add_argument("--conds", default="off,on", help="comma separated states, rotated per sample (off,on,a16)")
    ap.add_argument("--passive", action="store_true", help="no state switching: count only (default settings run)")
    ap.add_argument("--rot", type=int, default=0, help="rotation offset of the first sample")
    ap.add_argument("--extra-env", action="append", default=[], metavar="K=V", help="environment for the server")
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
    nonce_set = args.nonce_set or args.tag
    out_path = Path(args.out)

    names = args.conds.split(",")
    conds = {n: n for n in names}

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
    env["PYTHONPATH"] += ":" + str(HOOK_DIR)
    ctl_dir = base / "ctl"
    shutil.rmtree(ctl_dir, ignore_errors=True)
    ctl = Ctl(ctl_dir)
    env["OMLX_Q8_CTL"] = str(ctl_dir)
    if args.passive:
        env["OMLX_Q8_PASSIVE"] = "1"
    for item in args.extra_env:
        k, _, v = item.partition("=")
        env[k] = v

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
    res = {"schema": 2, "tag": args.tag, "conds": names, "lengths": args.lengths, "samples": args.samples,
           "rot": args.rot, "nonce_set": nonce_set,
           "design": "matched prompt per tuple, rotating condition order, salted prompts, prefix cache off, MTP off",
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
        ack = None if args.passive else ctl.call("set", state=cond)
        time.sleep(args.settle)
        before = ctl.call("snap")
        pw = m.power_state()
        rec = m.timed_prefill_strict(url, body, args.timeout)
        time.sleep(0.4)
        after = ctl.call("snap")
        pt = rec["prompt_tokens"]
        rec.update({"cond": cond, "phase": phase, "sample": i, "attempt": attempt, "length_requested": length,
                    "nonce": nonce, "t": time.time(), "set_gen": ack["gen"] if ack else None,
                    "state_ack": ack["state"] if ack else None,
                    "request_body_sha256": hashlib.sha256(body).hexdigest(), **ids_info,
                    "pp_tokens_per_s": (pt / rec["ttft_s"]) if (pt and rec["ttft_s"]) else None,
                    "power_before": pw["batt"].splitlines()[0] if pw["batt"] else "",
                    "therm_before": pw["therm"].strip().replace("\n", " | ")})
        if before is not None:
            diff = lambda key: {k: after[key].get(k, 0) - before[key].get(k, 0)  # noqa: E731
                                for k in set(after[key]) | set(before[key])
                                if after[key].get(k, 0) != before[key].get(k, 0)}
            rec["coverage"] = {"counts": diff("counts"), "proj": diff("proj"), "stock": diff("stock"),
                               "rows": diff("rows")}
            if "server_files" not in res:
                # The hook installs itself a moment after the first request has loaded the model.
                for _ in range(120):
                    if after["installed"] or "install_error" in after["meta"]:
                        break
                    time.sleep(0.5)
                    after = ctl.call("snap")
                meta = after["meta"]
                assert after["installed"] and "install_error" not in meta, meta
                assert meta["omlx_file"].startswith(args.tree + "/"), meta
                res["server_files"] = {k: meta.get(k) for k in ("omlx_file", "oa_file", "pid", "native", "mlp_classes")}
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
        if cov is not None and not args.passive:
            c = cov["counts"]
            if rec["cond"] == "on" and c.get("a8_proj", 0) == 0:
                out.append("Q8A8 did not run in the on condition")
            if rec["cond"] != "on" and (c.get("a8_proj", 0) or c.get("stage_a", 0)):  # off, off2, a16
                out.append(f"A8 ran in the {rec['cond']} condition: {c}")
            full_chunks = (rec.get("prompt_tokens") or 0) >= 2048
            if rec["cond"] == "a16" and full_chunks and c.get("q8_a16_native", 0) == 0:
                out.append("native Q8 A16 did not run in the a16 condition")
            if rec["cond"] in ("off", "off2") and c.get("q8_a16_native", 0):
                out.append("native Q8 A16 ran in the off condition")
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
                    k = (i + args.rot) % len(names)
                    order = names[k:] + names[:k]
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
                        f"a8={r.get('coverage', {}).get('counts', {}).get('a8_proj')}" for r in pair)
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
