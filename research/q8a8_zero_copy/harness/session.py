#!/usr/bin/env python3
"""One server session of the Qwen3.8-27B Q8 short-context decode benchmark (MTP off or on).

MTP is a load-time model setting, so each condition is its own server process. A round is two sessions
(off and on) in alternating order; both use the same prompts, so the pair is matched at the token level.
Q8A8 (qwen35_oq_a8_enabled, min_tokens 128) and every other setting are identical in both conditions;
only ``mtp_enabled`` differs. Prefix cache is disabled in the server settings.

Per request (stream, temperature 0, 128 tokens): TTFT, per-event timestamps and text, usage block
(backend generation_duration), MLX memory (peak reset before, peak/active/cache after, settled after a pause),
the MTP statistics line that the server logged for exactly this request (attributed by log byte offset),
power and thermal state. Results are written after every request.
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
import urllib.request
from pathlib import Path

HARNESS = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS))
import pp2_helpers as m  # noqa: E402  (copied from the earlier Q8A8 study)
from common import Ctl, sh  # noqa: E402

HOOK_DIR = Path(__file__).resolve().parent / "hook"
GiB = 1024 ** 3
MTP_HEAD = re.compile(r"MTP\[(?P<idx>\d+)\] finish=(?P<finish>\S+) tokens=(?P<tokens>\d+) cycles=(?P<cycles>\d+) "
                      r"tok/cycle=(?P<tpc>[\d.]+) accept=(?P<acc>\d+)/(?P<prop>\d+)")
MTP_EMITS = re.compile(r"emits\[init=(\d+),draft=(\d+),bonus=(\d+),verify=(\d+)\]")
MTP_TIMING = re.compile(r"timing\[backbone=([\d.]+)ms mtp=([\d.]+)ms sample=([\d.]+)ms cache=([\d.]+)ms\]")
MTP_ACTIVE = re.compile(r"MTP path activated for uid=(\d+) .*")
SUSPECT = re.compile(r"disabled|fallback|fall back|skipped|Traceback|ERROR|Error|WARNING", re.I)


def parse_mtp(text):
    """MTP summary lines of the batch generator (the scheduler echoes them once more; those are skipped)."""
    out = []
    for line in text.splitlines():
        if "mlx_lm_mtp.batch_generator" not in line or "MTP[" not in line or "finish=" not in line:
            continue
        mt = MTP_HEAD.search(line)
        rec = {"raw": line.split(" - INFO - ", 1)[-1]}
        if mt:
            g = mt.groupdict()
            rec.update({"finish": g["finish"], "tokens": int(g["tokens"]), "cycles": int(g["cycles"]),
                        "tok_per_cycle": float(g["tpc"]), "accepted": int(g["acc"]), "proposed": int(g["prop"])})
            dm = re.search(r"depth\[([^\]]*)\]", line)
            rec["depth"] = {k: [int(x) for x in v.split("/")] for k, v in
                            (kv.split("=") for kv in dm.group(1).split(","))} if dm else None
            for key, pat in (("d0", r" d0=(\d+)"), ):
                mm = re.search(pat, line)
                rec[key] = int(mm.group(1)) if mm else None
            cm = re.search(r"copy=(\d+)/(\d+) in (\d+)", line)
            rec["copy"] = [int(x) for x in cm.groups()] if cm else None
            em = MTP_EMITS.search(line)
            rec["emits"] = dict(zip(("init", "draft", "bonus", "verify"), (int(x) for x in em.groups()))) if em else None
            tm = MTP_TIMING.search(line)
            rec["timing_ms"] = dict(zip(("backbone", "mtp", "sample", "cache"), (float(x) for x in tm.groups()))) if tm else None
        out.append(rec)
    return out


def parse_active(text):
    return [l.split(" - INFO - ", 1)[-1] for l in text.splitlines() if MTP_ACTIVE.search(l)]


def ngram_stats(ids, n=3):
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return {"n": n, "total": len(grams), "distinct": len(set(grams)),
            "distinct_ratio": (len(set(grams)) / len(grams)) if grams else None}


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", required=True)
    ap.add_argument("--tree", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cond", required=True, choices=["off", "on"])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--arm", default="baseline", help="label of the arm (baseline, e2, ...)")
    ap.add_argument("--a8", choices=["on", "off"], default="on", help="qwen35_oq_a8_enabled model setting")
    ap.add_argument("--ref-main", default="models--Jundot--Qwen3.8-27B-oQ8e-mtp", help="HF cache dir name for provenance")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE passed to the server process (repeatable)")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048, 8192])
    ap.add_argument("--repeats", type=int, default=3, help="primary requests per length")
    ap.add_argument("--request-gap", type=float, default=0.0, help="extra seconds between requests (0 = settle_base rule)")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--settle-base", type=float, default=20.0, help="seconds before a request, plus 2 s per 1K tokens")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.max_tokens == 128, "this study is limited to 128 generated tokens"
    assert max(args.lengths) <= 32768, "this study is limited to 32K contexts"

    for k in [k for k in os.environ if k.startswith("OMLX_") or k == "PYTHONPATH"]:
        del os.environ[k]
    tag = f"{args.arm}_r{args.round}_mtp{args.cond}"
    base = Path(args.workdir) / f"sess_{tag}"
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
    model_settings = {"qwen35_oq_a8_enabled": args.a8 == "on", "qwen35_oq_a8_min_tokens": 128,
                      "mtp_enabled": args.cond == "on"}
    (base / "model_settings.json").write_text(json.dumps({"version": 1, "models": {args.model: model_settings}}, indent=2))
    ctl = Ctl(base / "ctl")
    env = {"PYTHONPATH": args.tree + ":" + str(HOOK_DIR), "HOME": str(base / "home"), "OMLX_DEC_CTL": str(base / "ctl")}
    for kv in args.env:
        k, v = kv.split("=", 1)
        env[k] = v

    probe = json.loads(subprocess.run(
        [args.python, "-c",
         "import json,sys,importlib.metadata as md,omlx;"
         "print(json.dumps({'omlx_file':omlx.__file__,'python':sys.version.split()[0],'mlx':md.version('mlx'),"
         "'mlx_lm':md.version('mlx-lm'),'mlx_vlm':md.version('mlx-vlm'),'tokenizers':md.version('tokenizers')}))"],
        capture_output=True, text=True, env={**os.environ, **env}, check=True).stdout.strip().splitlines()[-1])
    assert probe["omlx_file"].startswith(args.tree + "/"), probe
    gen_cfg = json.load(open(Path(args.checkpoint) / "generation_config.json"))
    shard = os.path.realpath(sorted(Path(args.checkpoint).glob("model-00001-of-*.safetensors"))[0])
    res = {"schema": 1, "cond": args.cond, "round": args.round, "tag": tag, "argv": sys.argv,
           "tree": args.tree, "tree_git_head": sh("git", "-C", args.tree, "rev-parse", "HEAD"),
           "tree_git_status": sh("git", "-C", args.tree, "status", "--porcelain"),
           "tree_git_branch": sh("git", "-C", args.tree, "branch", "--show-current"),
           "model_blob_shard1": shard, "model_ref_main": sh("cat", os.path.expanduser(
               f"~/.cache/huggingface/hub/{args.ref_main}/refs/main")),
           "versions": probe, "settings": settings, "model_settings": model_settings,
           "generation_config": gen_cfg, "env": env,
           "host": {"chip": sh("sysctl", "-n", "machdep.cpu.brand_string"), "hw_model": sh("sysctl", "-n", "hw.model"),
                    "memory_gib": int(sh("sysctl", "-n", "hw.memsize")) / GiB, "macos": sh("sw_vers", "-productVersion")},
           "requests": []}
    out_path = Path(args.out)

    def save():
        out_path.write_text(json.dumps(res, indent=1))

    tok = m.load_tokenizer(Path(args.checkpoint))
    flat = json.load(open(args.corpus))["tokens"]

    def build(length, nonce):
        """Prompt of exactly ``length`` tokens after the decode/encode round trip the server performs: token merges
        change the count slightly, so the id list is lengthened or shortened until the re-encoded count matches."""
        salt_ids = tok.encode(f"[decode sample {nonce}] ").ids
        off = int(hashlib.sha256(nonce.encode()).hexdigest(), 16) % len(flat)
        seq = salt_ids + (flat[off:] + flat[:off])[:length + 256 - len(salt_ids)]
        cut = length
        seen = set()
        for _ in range(40):
            prompt = tok.decode(seq[:cut])
            n = len(tok.encode(prompt).ids)
            if n == length or cut in seen:
                break
            seen.add(cut)
            cut += length - n
        return prompt, n

    port = m.free_port(8350)
    url = f"http://127.0.0.1:{port}"
    log_path = base / "logs" / "server.log"
    proc = m.start_server(args.python, base, Path(args.model_dir), port, env, log_path, [])

    def one_request(role, length, nonce, max_tokens, settle):
        prompt, prompt_ids = build(length, nonce)
        body = m.request_body(args.model, prompt, max_tokens)
        time.sleep(settle)
        pw = m.power_state()
        ctl.call("counters", reset=True)
        mem0 = ctl.call("mem", reset_peak=True)
        req = urllib.request.Request(url + "/v1/completions", data=body, method="POST", headers={
            "Content-Type": "application/json", "Accept": "text/event-stream"})
        t0 = time.perf_counter()
        events, usage, finish, text = [], {}, None, ""
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
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
                now = time.perf_counter() - t0
                if obj.get("usage"):
                    usage = obj["usage"]
                for ch in obj.get("choices") or []:
                    if ch.get("finish_reason"):
                        finish = ch["finish_reason"]
                    if ch.get("text"):
                        events.append({"t": now, "text": ch["text"]})
                        text += ch["text"]
        total = time.perf_counter() - t0
        time.sleep(1.0)
        mem1 = ctl.call("mem")
        cnt1 = ctl.call("counters")
        time.sleep(4.0)
        mem2 = ctl.call("mem")
        time.sleep(10.0)
        mem3 = ctl.call("mem")
        cnt2 = ctl.call("counters")
        life = ctl.call("lifetime")["stats"]
        prepared_scan = ctl.call("scan_prepared")
        out_ids = tok.encode(text).ids
        first_text = events[0]["text"] if events else ""
        k1 = len(tok.encode(first_text).ids) if first_text else 0
        n = usage.get("completion_tokens")
        gd = usage.get("generation_duration")
        rec = {"role": role, "length": length, "nonce": nonce, "max_tokens": max_tokens,
               "request_body_sha256": hashlib.sha256(body).hexdigest(), "planned_prompt_ids": prompt_ids,
               "usage": usage, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": n,
               "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
               "finish_reason": finish, "client_ttft_s": events[0]["t"] if events else None,
               "client_total_s": total, "n_events": len(events), "events": events,
               "first_event_tokens_est": k1, "text": text, "output_ids_est": out_ids[:200],
               "ngram3": ngram_stats(out_ids), "mem_before": mem0, "mem_after": mem1, "mem_settled": mem2, "mem_settled_15s": mem3,
               "counters_1s": cnt1["counts"], "counters_15s": cnt2["counts"], "rows_hist": cnt2.get("rows_hist"), "clear_events": cnt2["clear_events"],
               "lifetime_stats": life,
               "prepared_scan": {k: prepared_scan.get(k) for k in ("layers_with_prepared_copy", "prepared_copy_bytes")},
               "power_before": pw["batt"].splitlines()[0] if pw["batt"] else "",
               "therm_before": pw["therm"].strip().replace("\n", " | "), "t_wall": time.time()}
        if rec["client_ttft_s"]:
            rec["prefill_tps_ttft"] = rec["prompt_tokens"] / rec["client_ttft_s"]
        # throughput variants
        if n and gd:
            rec["tg_backend_all"] = n / gd                      # includes the first token (server definition)
            rec["tg_backend_after_first_event"] = (n - k1) / gd  # decode only, first emitted tokens excluded
        if len(events) >= 2 and n:
            span = events[-1]["t"] - events[0]["t"]
            rec["tg_stream_after_first_event"] = (n - k1) / span if span > 0 else None
        ok = []
        if n != max_tokens:
            ok.append(f"completion_tokens={n}")
        if finish != "length":
            ok.append(f"finish_reason={finish}")
        if rec["prompt_tokens"] != length:
            ok.append(f"prompt_tokens={rec['prompt_tokens']} != {length}")
        if rec["cached_tokens"]:
            ok.append(f"cached_tokens={rec['cached_tokens']}")
        if rec["planned_prompt_ids"] != length:
            ok.append(f"tokenizer round trip gives {rec['planned_prompt_ids']}")
        rec["problems"] = ok
        rec["valid"] = not ok
        res["requests"].append(rec)
        save()
        print(f"[{tag}] {role:<7} L={length:>6} n={n} ttft={rec['client_ttft_s'] and round(rec['client_ttft_s'], 3)} "
              f"tg_be={rec.get('tg_backend_after_first_event') and round(rec['tg_backend_after_first_event'], 2)} "
              f"peak={mem1['peak'] / GiB:.2f} act={mem3['active'] / GiB:.2f} phys={mem3['phys_footprint'] / GiB:.2f} "
              f"a8={cnt2['counts'].get('a8_qmm_calls', 0)} prep={cnt2['counts'].get('prepare_builds', 0)} "
              f"{ok if ok else ''}", flush=True)
        return rec

    def finalize():
        """The server writes its log file with a delay, so MTP statistics are attributed after the process has
        exited: request k of the session (0 = the 1-token load request) is uid k, and the line is accepted only
        when its generated and primed token counts match that request's usage block."""
        txt = log_path.read_text(errors="replace")
        summ = {}
        for rec in parse_mtp(txt):
            mm = re.match(r"MTP\[(\d+)\]", rec["raw"])
            summ[int(mm.group(1))] = rec
        acts = {}
        for line in parse_active(txt):
            mm = re.search(r"uid=(\d+) .*primed=(\d+)", line)
            if mm:
                acts[int(mm.group(1))] = {"primed": int(mm.group(2)), "raw": line}
        res["mtp_summary_lines_total"] = len(summ)
        res["mtp_activation_lines_total"] = len(acts)
        res["log_suspect_lines"] = [l[:300] for l in txt.splitlines() if SUSPECT.search(l)][:40]
        res["log_lines_penalty_or_depth"] = [l[:300] for l in txt.splitlines()
                                             if re.search(r"penalt|fixed_depth|Lightning MTP|MTP.*(enabled|disabled)|oQ A8", l, re.I)][:20]
        for k, rec in enumerate(res["requests"], start=1):
            line, act = summ.get(k), acts.get(k)
            rec["uid"] = k
            rec["mtp_log"] = [line] if line else []
            rec["mtp_activation"] = act
            if args.cond == "on":
                if not line or not act:
                    rec["problems"].append(f"missing MTP evidence for uid {k} (summary={bool(line)}, activation={bool(act)})")
                else:
                    if line.get("tokens") != rec["completion_tokens"]:
                        rec["problems"].append(f"MTP summary tokens={line.get('tokens')} != completion_tokens")
                    if abs(act["primed"] - rec["prompt_tokens"]) > 1:
                        rec["problems"].append(f"MTP primed={act['primed']} vs prompt_tokens={rec['prompt_tokens']}")
                    if line.get("cycles", 0) <= 0 or line.get("proposed", 0) <= 0:
                        rec["problems"].append("MTP without draft cycles")
                    tm = line.get("timing_ms")
                    if tm:
                        rec["mtp_log_gen_s"] = sum(tm.values()) / 1000.0
            elif line or act:
                rec["problems"].append("MTP activity logged in the off condition")
            rec["valid"] = not rec["problems"]

    try:
        m.wait_ready(url + "/v1/models", proc)
        res["power_start"] = m.power_state()
        # first request loads the model and installs the engine; 1 token only
        prompt, _ = build(32, f"load-{tag}")  # short: loads the model without entering the A8 route
        req = urllib.request.Request(url + "/v1/completions", data=m.request_body(args.model, prompt, 1), method="POST",
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=args.timeout).read()
        for _ in range(60):
            fl = ctl.call("flags")
            if args.a8 == "off" or (fl["mlp_patched"] and fl["gdn_registered"]):
                break
            time.sleep(0.5)
        res["flags"] = fl
        if args.a8 == "on":
            assert fl["mlp_patched"] and fl["gdn_registered"], f"Q8A8 patch is not wired: {fl}"
        assert args.a8 == "off" or fl["oq_a8_available"] is True, f"native A8 kernel unavailable: {fl}"
        assert fl["omlx_file"].startswith(args.tree + "/"), fl
        assert args.a8 == "off" or fl["ext_file"].startswith(args.tree + "/"), fl
        installed = ctl.call("install")["installed"]
        assert installed or args.a8 == "off", "counter hook could not be installed"
        res["mem_after_load"] = ctl.call("mem", reset_peak=True)
        time.sleep(5.0)
        res["mem_after_load_5s"] = ctl.call("mem")
        # warm-up: every context, full 128 tokens, nonce shared by both conditions of the round
        for L in args.lengths:
            one_request("warmup", L, f"dec-warm-{args.round}-{L}", args.max_tokens, 5.0)
        time.sleep(10.0)
        res["mem_after_warmup"] = ctl.call("mem", reset_peak=True)
        for rep_i in range(args.repeats):
            for L in args.lengths:
                one_request("primary", L, f"dec-r{args.round}-{L}-p{rep_i}", args.max_tokens,
                            args.request_gap or (args.settle_base + 2.0 * L / 1024))
        res["power_end"] = m.power_state()
    finally:
        m.stop_server(proc)
        finalize()
        save()
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
