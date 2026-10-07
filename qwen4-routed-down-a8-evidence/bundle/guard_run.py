#!/usr/bin/env python3
"""Run one model process under a memory guard (the Flash-Next process needs ~110 GB of a 128 GB Mac).

On 2026-10-07 a long run with free pages at ~0 and the compressor churning hundreds of GB made the
kernel panic (watchdog timeout) and reboot the machine. This launcher exists so that never needs a
reboot again, which matters when the machine is only reachable remotely:

  * pre-flight: refuses to start when another model/harness process is alive, when the compressor
    or swap is already in use, or when free + inactive + speculative memory is below the model need;
  * while the job runs it samples every second and kills the whole process group (SIGTERM, then
    SIGKILL) when (a) swap exceeds --max-swap-gib or the compressor exceeds --max-compressor-gib for
    --strikes consecutive samples, or (b) after --grace-secs (model load: the OS compresses a lot of
    memory while 100 GB are read, which is normal) the decompression rate stays above
    --max-decompress-pps pages/s for --thrash-secs, which is the signature of the thrashing that
    preceded the panic (a healthy, resident model decompresses almost nothing);
  * every sample goes to <log>.guard.json (atomic replace), the job output goes to <log> unbuffered
    to disk, and the exit reason is written to <log>.exit.json.

Exit code: the job's own, 90 for a pre-flight refusal, 91 when the guard killed the job.

    guard_run.py --log logs/x.log -- <python> -u script.py ...
"""
import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time

PAGE = 16384
GiB = 1024 ** 3
HARNESS = ("quality_down.py", "capture_real.py", "memory_down.py", "measure_down_alt.py", "omlx-server",
           "microbench_down.py", "oracle_down.py", "probe_meta_ceiling.py", "quality_prod.py", "memory_prod.py",
           "measure_prod.py", "smoke_tools.py", "omlx.cli")


def vm():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    g = lambda k: int(re.search(rf"{k}:\s+(\d+)", out).group(1)) * PAGE / GiB
    return {"free": g("Pages free"), "inactive": g("Pages inactive"), "speculative": g("Pages speculative"),
            "compressor": g("Pages occupied by compressor"), "wired": g("Pages wired down")}


def decompressions():
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    return int(re.search(r"Decompressions:\s+(\d+)", out).group(1))


def swap_gib():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used = ([\d.]+)M", out)
    return float(m.group(1)) / 1024 if m else 0.0


def other_model_processes():
    out = subprocess.run(["ps", "-axo", "pid,stat,command"], capture_output=True, text=True).stdout
    me = os.getpid()
    hits = []
    for line in out.splitlines()[1:]:
        pid, stat, cmd = line.split(None, 2)
        if int(pid) != me and any(h in cmd for h in HARNESS) and "guard_run.py" not in cmd:
            hits.append((pid, stat, cmd[:100]))
    return hits


def descendants(root):
    out = subprocess.run(["ps", "-axo", "pid,ppid"], capture_output=True, text=True).stdout
    kids = {}
    for line in out.splitlines()[1:]:
        pid, ppid = (int(v) for v in line.split())
        kids.setdefault(ppid, []).append(pid)
    found, todo = [], [root]
    while todo:
        for c in kids.get(todo.pop(), []):
            found.append(c)
            todo.append(c)
    return found


def kill_tree(root, sig):
    """The job may start servers in their own session, so a group kill alone would miss them."""
    for pid in descendants(root) + [root]:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    # Loading the 99 GiB Flash-Next checkpoint on a 128 GiB Mac drives the compressor to 60-68 GiB for a few
    # seconds even when everything is healthy (every successful run of this study peaked there; the 56 GiB
    # default killed a healthy model load on 2026-10-07). The swap limit and the thrash detector (sustained
    # decompression after the load grace) stay the real guards.
    ap.add_argument("--max-compressor-gib", type=float, default=100.0)
    ap.add_argument("--grace-secs", type=float, default=300.0)
    ap.add_argument("--max-decompress-pps", type=float, default=5000.0)
    ap.add_argument("--thrash-secs", type=float, default=60.0)
    ap.add_argument("--max-swap-gib", type=float, default=4.0)
    # Swap that an idle process left behind is not thrashing; the kill limit above and the decompression
    # detector remain the guards while the job runs.
    ap.add_argument("--max-start-swap-gib", type=float, default=2.0, help="swap already in use at start")
    ap.add_argument("--min-available-gib", type=float, default=100.0, help="free+inactive+speculative at start")
    ap.add_argument("--strikes", type=int, default=3)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--no-preflight", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        ap.error("no command")
    status_path, exit_path = a.log + ".guard.json", a.log + ".exit.json"

    def write(path, obj):
        with open(path + ".tmp", "w") as f:
            json.dump(obj, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(path + ".tmp", path)

    s0, sw0 = vm(), swap_gib()
    if not a.no_preflight:
        why = []
        others = other_model_processes()
        if others:
            why.append(f"other model/harness processes alive: {others}")
        if s0["compressor"] > 2.0:
            why.append(f"compressor already {s0['compressor']:.1f} GiB (> 2)")
        if sw0 > a.max_start_swap_gib:
            why.append(f"swap already {sw0:.1f} GiB (> {a.max_start_swap_gib:g})")
        avail = s0["free"] + s0["inactive"] + s0["speculative"]
        if avail < a.min_available_gib:
            why.append(f"available {avail:.0f} GiB < {a.min_available_gib:.0f}")
        if why:
            write(exit_path, {"result": "preflight_refused", "reasons": why, "vm": s0, "swap_gib": sw0})
            print("PREFLIGHT REFUSED:", *why, sep="\n  ", file=sys.stderr)
            return 90

    logf = os.open(a.log, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True,
                         env=dict(os.environ, PYTHONUNBUFFERED="1"))
    strikes, peak_c, peak_s, reason = 0, 0.0, 0.0, None
    hist = []  # (time, decompressions)
    thrash_since, rate = None, 0.0
    while p.poll() is None:
        s, sw = vm(), swap_gib()
        now = time.time()
        hist.append((now, decompressions()))
        while len(hist) > 1 and now - hist[1][0] >= 10:
            hist.pop(0)
        rate = (hist[-1][1] - hist[0][1]) / max(hist[-1][0] - hist[0][0], 1e-6)
        if now - t0 > a.grace_secs and rate > a.max_decompress_pps:
            thrash_since = thrash_since or now
        else:
            thrash_since = None
        peak_c, peak_s = max(peak_c, s["compressor"]), max(peak_s, sw)
        bad = s["compressor"] > a.max_compressor_gib or sw > a.max_swap_gib
        strikes = strikes + 1 if bad else 0
        write(status_path, {"t": time.time(), "elapsed_s": time.time() - t0, "pid": p.pid, "vm": s,
                            "swap_gib": sw, "peak_compressor_gib": peak_c, "peak_swap_gib": peak_s,
                            "strikes": strikes, "decompress_pages_per_s": rate,
                            "thrash_for_s": (now - thrash_since) if thrash_since else 0})
        if thrash_since and now - thrash_since >= a.thrash_secs:
            strikes = a.strikes
            bad = True
        if strikes >= a.strikes:
            reason = (f"compressor {s['compressor']:.1f} GiB (limit {a.max_compressor_gib}) / "
                      f"swap {sw:.1f} GiB (limit {a.max_swap_gib}) / "
                      f"decompress {rate:.0f} pages/s for {(now - thrash_since) if thrash_since else 0:.0f}s "
                      f"(limit {a.max_decompress_pps:.0f})")
            tree = descendants(p.pid)
            kill_tree(p.pid, signal.SIGTERM)
            time.sleep(8)
            for pid in tree + [p.pid]:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            break
        time.sleep(a.interval)
    rc = p.wait() if reason is None else (p.poll() if p.poll() is not None else -9)
    write(exit_path, {"result": "guard_killed" if reason else "finished", "reason": reason, "returncode": rc,
                      "elapsed_s": time.time() - t0, "peak_compressor_gib": peak_c, "peak_swap_gib": peak_s,
                      "start_vm": s0})
    if reason:
        print("GUARD KILLED THE JOB:", reason, file=sys.stderr)
        return 91
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
