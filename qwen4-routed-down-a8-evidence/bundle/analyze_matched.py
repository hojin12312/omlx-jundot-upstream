#!/usr/bin/env python3
"""Aggregate ``measure_prod.py`` receipts (schema 2).

  analyze_matched.py paired  RUN.json [RUN.json ...]   paired b0/b1 (or A/A) runs
  analyze_matched.py control RUN.json [RUN.json ...]   single-condition runs matched by request bytes

Method (fixed before the runs, see PREREGISTRATION.md):
  * a sample is valid when its pair carried identical request bytes and prompt ids, a first token was
    seen, usage.prompt_tokens exists and is equal inside the pair, and (where reported) cached_tokens
    is 0. Invalid pairs are listed and excluded, never repaired.
  * PP/s = usage.prompt_tokens / client TTFT; the table value of a condition is the arithmetic mean.
  * gain = mean over matched pairs of (B1 PP/s / B0 PP/s - 1); the interval is the Student-t 95%
    interval of the pair ratios. Per-run and per-order results are reported next to it.
  * TTFT medians are medians of the measured TTFT, not recomputed from PP/s.
"""
import collections
import hashlib
import json
import statistics as st
import sys

try:
    from scipy import stats as _sps
except ImportError:  # pragma: no cover
    _sps = None
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
        20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069}
LABEL = {16384: "16K", 32768: "32K", 65536: "64K", 131072: "128K", 8192: "8K"}


def tcrit(df):
    return float(_sps.t.ppf(0.975, df)) if _sps else T975.get(df, 1.96)


def ci(vals):
    """mean, half width of the t interval (both of the raw values)."""
    n = len(vals)
    if n < 2:
        return st.mean(vals), float("nan")
    return st.mean(vals), tcrit(n - 1) * st.stdev(vals) / n ** 0.5


def load(paths):
    runs = []
    for p in paths:
        d = json.load(open(p))
        assert d.get("schema") == 2, f"{p}: not a schema 2 receipt"
        d["_path"] = p
        runs.append(d)
    return runs


def timed(d):
    return [r for r in d["measurements"] if r["phase"] == "timed"]


def validity(d):
    recs = timed(d)
    bad = [r for r in recs if not r["valid"]]
    return len(recs), len(bad), collections.Counter(x for r in bad for x in r.get("invalid_reasons", []))


def pairs_of(d):
    """{(length, sample): {cond: record}} of valid pairs, plus the cond that ran first."""
    out, first = collections.defaultdict(dict), {}
    for r in sorted((r for r in timed(d) if r["valid"]), key=lambda r: r["t"]):
        key = (r["length_requested"], r["sample"])
        out[key][r["cond"]] = r
        first.setdefault(key, r["cond"])
    return out, first


def coverage_line(recs):
    if not recs or "coverage" not in recs[0]:
        return "no counters (hook off)"
    c = lambda f: sorted({f(r) for r in recs})  # noqa: E731
    down = c(lambda r: r["coverage"]["counts"].get("down_a8_results", 0))
    gu = c(lambda r: r["coverage"]["counts"].get("gate_up_a8_results", 0))
    dl = c(lambda r: r["coverage"]["launch"].get("down:G10:mode2", 0))
    gl = c(lambda r: r["coverage"]["launch"].get("gate_up:G40:mode1", 0))
    mx = max((int(k) for r in recs for k in r["coverage"]["down_rows"]), default=0)
    return (f"down results/req {down}  down launches (G10 mode2) {dl}  gate_up results {gu}  "
            f"gate_up launches (G40 mode1) {gl}  max rows per Down call {mx}")


def paired(runs):
    for d in runs:
        n, bad, why = validity(d)
        print(f"{d['_path']}: tag={d['tag']} mode={d['mode']} first={d['first']} nonce_set={d['nonce_set']} "
              f"tree={d['tree_git_head'][:8]} timed={n} invalid={bad} {dict(why)}")
    lengths = sorted({r["length_requested"] for d in runs for r in timed(d)})
    rows, receipt = [], {"runs": [d["_path"] for d in runs], "lengths": {}}
    for L in lengths:
        allp, per_run = [], []
        for d in runs:
            pr, first = pairs_of(d)
            rs = []
            for (l, i), s in sorted(pr.items()):
                if l != L or "b0" not in s or "b1" not in s:
                    continue
                b0, b1 = s["b0"], s["b1"]
                assert b0["request_body_sha256"] == b1["request_body_sha256"]
                assert b0["prompt_ids_sha256"] == b1["prompt_ids_sha256"]
                assert b0["prompt_tokens"] == b1["prompt_tokens"]
                ratio = b1["pp_tokens_per_s"] / b0["pp_tokens_per_s"]
                rs.append(ratio)
                allp.append({"run": d["tag"], "sample": i, "nonce": b0["nonce"], "first": first[(l, i)],
                             "prompt_tokens": b0["prompt_tokens"], "body_sha256": b0["request_body_sha256"][:16],
                             "ids_sha256": b0["prompt_ids_sha256"][:16], "ratio": ratio,
                             "b0_pp": b0["pp_tokens_per_s"], "b1_pp": b1["pp_tokens_per_s"],
                             "b0_ttft": b0["ttft_s"], "b1_ttft": b1["ttft_s"],
                             "b0_cov": b0.get("coverage"), "b1_cov": b1.get("coverage")})
            if rs:
                per_run.append((d["tag"], 100 * (st.mean(rs) - 1), len(rs)))
        if not allp:
            continue
        ratios = [x["ratio"] for x in allp]
        mean_ratio, half = ci(ratios)
        lo, hi = 100 * (mean_ratio - half - 1), 100 * (mean_ratio + half - 1)
        order = {k: [x["ratio"] for x in allp if x["first"] == k] for k in ("b0", "b1")}
        o = lambda v: f"{100 * (st.mean(v) - 1):+.2f}% (n={len(v)})" if v else "n/a"  # noqa: E731
        pt = sorted({x["prompt_tokens"] for x in allp})
        b0pp, b1pp = st.mean(x["b0_pp"] for x in allp), st.mean(x["b1_pp"] for x in allp)
        t0, t1 = st.median(x["b0_ttft"] for x in allp), st.median(x["b1_ttft"] for x in allp)
        b0cov = [x["b0_cov"] for x in allp if x["b0_cov"]]
        b1cov = [x["b1_cov"] for x in allp if x["b1_cov"]]
        print(f"\n== {LABEL.get(L, L)}  pairs={len(allp)}  prompt tokens {pt[0]}..{pt[-1]}")
        print(f"   PP/s b0 {b0pp:.1f}  b1 {b1pp:.1f}   gain {100 * (mean_ratio - 1):+.2f}%  95% t-CI "
              f"[{lo:+.2f}%, {hi:+.2f}%]  min/max {100 * (min(ratios) - 1):+.2f}%/{100 * (max(ratios) - 1):+.2f}%  "
              f"b1 faster in {sum(r > 1 for r in ratios)}/{len(ratios)}")
        print(f"   per run: {', '.join(f'{t} {g:+.2f}% (n={n})' for t, g, n in per_run)}   "
              f"by order: b0 first {o(order['b0'])}, b1 first {o(order['b1'])}")
        print(f"   median TTFT b0 {t0:.2f} s  b1 {t1:.2f} s   (mean {st.mean(x['b0_ttft'] for x in allp):.2f} / "
              f"{st.mean(x['b1_ttft'] for x in allp):.2f})")
        print(f"   coverage b0: {coverage_line([{'coverage': c} for c in b0cov])}")
        print(f"   coverage b1: {coverage_line([{'coverage': c} for c in b1cov])}")
        gain_s = f"{100 * (mean_ratio - 1):+.2f}%"
        rows.append((L, f"| {LABEL.get(L, L)} | {b0pp:.1f} | {b1pp:.1f} | {gain_s} [{lo:+.2f}%, {hi:+.2f}%] |", t0, t1))
        receipt["lengths"][str(L)] = {"pairs": allp, "gain_pct": 100 * (mean_ratio - 1), "ci_pct": [lo, hi],
                                      "b0_pp_mean": b0pp, "b1_pp_mean": b1pp, "b0_ttft_median_s": t0,
                                      "b1_ttft_median_s": t1, "per_run_gain_pct": per_run,
                                      "by_order_gain_pct": {k: 100 * (st.mean(v) - 1) if v else None
                                                            for k, v in order.items()}}
    print("\nPERF_ROWS:")
    for _, row, _, _ in rows:
        print(row)
    for L, _, t0, t1 in rows:
        print(f"TTFT {LABEL.get(L, L)}: A16 {t0:.2f} s, A8 {t1:.2f} s")
    return receipt


def control(runs):
    labels = []
    for d in runs:
        n, bad, why = validity(d)
        print(f"{d['_path']}: label={d['label']} hook={d['hook']} tree={d['tree_git_head'][:8]} "
              f"env_down={d['env'].get('OMLX_M5_ROUTED_DOWN_A8')} nonce_set={d['nonce_set']} timed={n} invalid={bad}")
        if d["label"] not in labels:
            labels.append(d["label"])
    lengths = sorted({r["length_requested"] for d in runs for r in timed(d)})
    receipt = {"runs": [d["_path"] for d in runs], "lengths": {}}
    for L in lengths:
        # {(sample, body hash): {label: [records over sessions]}}
        grid = collections.defaultdict(lambda: collections.defaultdict(list))
        for d in runs:
            for r in timed(d):
                if r["valid"] and r["length_requested"] == L:
                    grid[(r["sample"], r["request_body_sha256"])][d["label"]].append(r)
        trip = {k: v for k, v in grid.items() if all(lbl in v for lbl in labels)}
        print(f"\n== {LABEL.get(L, L)}  matched triplets (same request bytes under every label): {len(trip)}")
        mean_pp = {lbl: st.mean(st.mean(r["pp_tokens_per_s"] for r in v[lbl]) for v in trip.values())
                   for lbl in labels}
        med_ttft = {lbl: st.median(st.mean(r["ttft_s"] for r in v[lbl]) for v in trip.values()) for lbl in labels}
        print("   PP/s " + "  ".join(f"{lbl} {mean_pp[lbl]:.1f}" for lbl in labels) +
              "   median TTFT " + "  ".join(f"{lbl} {med_ttft[lbl]:.2f}s" for lbl in labels))
        rec = {"triplets": len(trip), "pp_mean": mean_pp, "ttft_median": med_ttft, "ratios": {}}
        for a, b in ((labels[i], labels[j]) for i in range(len(labels)) for j in range(len(labels)) if i < j):
            ratios = [st.mean(r["pp_tokens_per_s"] for r in v[b]) / st.mean(r["pp_tokens_per_s"] for r in v[a])
                      for v in trip.values()]
            m_, h_ = ci(ratios)
            print(f"   {b}/{a}: {100 * (m_ - 1):+.2f}%  95% t-CI [{100 * (m_ - h_ - 1):+.2f}%, "
                  f"{100 * (m_ + h_ - 1):+.2f}%]  n={len(ratios)}")
            rec["ratios"][f"{b}/{a}"] = {"gain_pct": 100 * (m_ - 1), "ci_pct": [100 * (m_ - h_ - 1), 100 * (m_ + h_ - 1)],
                                         "n": len(ratios)}
        for lbl in labels:
            recs = [r for d in runs if d["label"] == lbl for r in timed(d) if r["valid"] and r["length_requested"] == L]
            print(f"   coverage {lbl}: {coverage_line(recs)}")
        receipt["lengths"][str(L)] = rec
    print("\nREFERENCE_CONTROL_ROWS:")
    for L in lengths:
        r = receipt["lengths"][str(L)]
        print(f"| {LABEL.get(L, L)} | " + " | ".join(f"{r['pp_mean'][lbl]:.1f}" for lbl in labels) + " |")
    return receipt


if __name__ == "__main__":
    kind, paths = sys.argv[1], sys.argv[2:]
    runs = load(paths)
    out = paired(runs) if kind == "paired" else control(runs)
    out["analysis_sha256_of_inputs"] = {p: hashlib.sha256(open(p, "rb").read()).hexdigest() for p in paths}
    print("\nreceipt keys:", list(out))
    json.dump(out, open(paths[0].rsplit(".json", 1)[0] + f".{kind}.analysis.json", "w"), indent=1, default=str)
