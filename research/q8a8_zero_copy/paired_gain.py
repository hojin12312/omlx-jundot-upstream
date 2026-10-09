"""Per-request paired gains: the same prompt (nonce) in two arms of the same round."""
import glob, json, statistics as st, sys, collections
d = sys.argv[1]
def pairs(prefix, mtp, a, b):
    rec = collections.defaultdict(dict)
    for f in glob.glob(f"{d}/{prefix}_*_r[0-9]_mtp{mtp}.json"):
        j = json.load(open(f)); arm = j["tag"].split("_r")[0].removeprefix(prefix + "_")
        for r in j["requests"]:
            if r["role"] == "primary" and r["client_ttft_s"] and r["prompt_tokens"] == r["length"] and not r["cached_tokens"]:
                rec[(r["length"], r["nonce"])][arm] = r["prompt_tokens"] / r["client_ttft_s"]
    out = collections.defaultdict(list)
    for (L, _), v in rec.items():
        if a in v and b in v:
            out[L].append(v[a] / v[b] - 1)
    return out
print("| Context | pairs | native vs OFF median (min..max) | pairs | native vs transposed median (min..max) | transposed vs OFF median (min..max) |")
print("|---:|---:|---|---:|---|---|")
for title, sets in (("MTP off", [("m1", "off"), ("m1L", "off")]), ("MTP on", [("m1M", "on")])):
    print(f"| **{title}** | | | | | |")
    X = {}
    for prefix, mtp in sets:
        for a, b, key in (("native", "off", "no"), ("native", "transposed", "nt"), ("transposed", "off", "to")):
            for L, v in pairs(prefix, mtp, a, b).items():
                X.setdefault(L, {})[key] = v
    f = lambda v: f"{st.median(v)*100:+.1f}% ({min(v)*100:+.1f}..{max(v)*100:+.1f})"
    for L in sorted(X):
        print(f"| {L} | {len(X[L]['no'])} | {f(X[L]['no'])} | {len(X[L]['nt'])} | {f(X[L]['nt'])} | {f(X[L]['to'])} |")
