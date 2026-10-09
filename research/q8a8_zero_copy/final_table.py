"""The table requested for the final report: per context, prefill tok/s of the three arms, improvements, TTFT, memory."""
import glob, json, statistics as st, sys, collections
d = sys.argv[1]; GIB = 2**30
def load(prefix, mtp):
    cell = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(f"{d}/{prefix}_*_r[0-9]_mtp{mtp}.json")):
        j = json.load(open(f)); arm = j["tag"].split("_r")[0].removeprefix(prefix + "_")
        for r in j["requests"]:
            if r["role"] == "primary" and r["client_ttft_s"] and r["prompt_tokens"] == r["length"] and not r["cached_tokens"]:
                cell[r["length"]][arm].append(r)
    return cell
med = lambda x: st.median(x)
for title, sets in (("MTP off", [("m1", "off"), ("m1L", "off")]), ("MTP on", [("m1M", "on")])):
    print(f"### {title}\n")
    print("| Context | OFF tok/s | Transposed tok/s | Native tok/s | Native vs OFF | Native vs Transposed | TTFT OFF / Transposed / Native (s) | Peak active OFF / Transposed / Native (GiB) | Settled active OFF / Transposed / Native (GiB) | phys_footprint OFF / Transposed / Native (GiB) |")
    print("|---:|---:|---:|---:|---:|---:|---|---|---|---|")
    for prefix, mtp in sets:
        c = load(prefix, mtp)
        for L in sorted(c):
            a = {k: c[L][k] for k in ("off", "transposed", "native")}
            tps = {k: med([r["prompt_tokens"] / r["client_ttft_s"] for r in v]) for k, v in a.items()}
            tt = {k: med([r["client_ttft_s"] for r in v]) for k, v in a.items()}
            pk = {k: med([r["mem_after"]["peak"] for r in v]) / GIB for k, v in a.items()}
            ac = {k: med([r["mem_settled_15s"]["active"] for r in v]) / GIB for k, v in a.items()}
            ph = {k: med([r["mem_settled_15s"]["phys_footprint"] for r in v]) / GIB for k, v in a.items()}
            n = {k: len(v) for k, v in a.items()}
            print(f"| {L} (n={n['native']}) | {tps['off']:.0f} | {tps['transposed']:.0f} | {tps['native']:.0f} | "
                  f"{(tps['native']/tps['off']-1)*100:+.1f}% | {(tps['native']/tps['transposed']-1)*100:+.1f}% | "
                  f"{tt['off']:.3f} / {tt['transposed']:.3f} / {tt['native']:.3f} | "
                  f"{pk['off']:.2f} / {pk['transposed']:.2f} / {pk['native']:.2f} | "
                  f"{ac['off']:.2f} / {ac['transposed']:.2f} / {ac['native']:.2f} | {ph['off']:.2f} / {ph['transposed']:.2f} / {ph['native']:.2f} |")
    print()
