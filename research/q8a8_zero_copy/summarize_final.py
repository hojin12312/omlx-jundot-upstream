"""Three-arm summary: A8 off (stock), A8 group-major ("transposed"), A8 checkpoint-layout ("native").

usage: summarize_final.py RESULTS_DIR PREFIX MTP
Per context length: medians over the primary requests of all rounds, the spread (min..max) and the improvement
ratios, with a paired-by-round check (each round's median of one arm against the same round's median of the other).
"""
import glob, json, statistics as st, sys, collections
GIB = 2**30
d, prefix, mtp = sys.argv[1], sys.argv[2], sys.argv[3]
ARMS = ["off", "transposed", "native"]
cell = collections.defaultdict(lambda: collections.defaultdict(list))   # (L) -> arm -> [(round, rec)]
ses = {}
for f in sorted(glob.glob(f"{d}/{prefix}_*_r[0-9]_mtp{mtp}.json")):
    j = json.load(open(f))
    arm = j["tag"].split("_r")[0].removeprefix(prefix + "_")
    ses[(arm, j["round"])] = j
    for r in j["requests"]:
        # prefill metrics need a first token and the planned prompt with no cached tokens; a request that hit EOS early
        # still has a valid TTFT, so only the decode figure is restricted to full 128-token generations
        if r["role"] == "primary" and r["client_ttft_s"] and r["prompt_tokens"] == r["length"] and not r["cached_tokens"]:
            cell[r["length"]][arm].append((j["round"], r))

def med(x):
    x = [v for v in x if v is not None]
    return st.median(x) if x else float("nan")

def rng(x):
    return f"{min(x):.3f}..{max(x):.3f}" if x else "-"

print(f"## {prefix} MTP {mtp}: sessions " + ", ".join(f"{a}:r{sorted(r for (aa, r) in ses if aa == a)}" for a in ARMS))
print("| L | arm | n | TTFT s (median, min..max) | prefill tok/s | decode tok/s | peak GiB | settled active GiB | phys GiB |")
print("|---:|---|---:|---|---:|---:|---:|---:|---:|")
summary = {}
for L in sorted(cell):
    for a in ARMS:
        rs = [r for _, r in cell[L].get(a, [])]
        if not rs:
            continue
        t = [r["client_ttft_s"] for r in rs]
        summary[(L, a)] = {"ttft": med(t), "tps": med([r["prefill_tps_ttft"] for r in rs]), "t": t}
        print(f"| {L} | {a} | {len(rs)} | {med(t):.3f} ({rng(t)}) | {med([r['prefill_tps_ttft'] for r in rs]):.0f} | "
              f"{med([r.get('tg_backend_after_first_event') for r in rs if r['completion_tokens'] == 128]):.2f} | "
              f"{med([r['mem_after']['peak'] for r in rs])/GIB:.3f} | {med([r['mem_settled_15s']['active'] for r in rs])/GIB:.3f} | "
              f"{med([r['mem_settled_15s']['phys_footprint'] for r in rs])/GIB:.3f} |")
print()
print("| L | native vs off (TTFT) | native vs transposed (TTFT) | transposed vs off | paired rounds native/transposed | paired rounds native/off | spread-separated vs transposed |")
print("|---:|---:|---:|---:|---|---|---|")
def paired(L, a, b):
    out = []
    for rd in sorted({r for r, _ in cell[L].get(a, [])} & {r for r, _ in cell[L].get(b, [])}):
        x = med([r["client_ttft_s"] for q, r in cell[L][a] if q == rd]); y = med([r["client_ttft_s"] for q, r in cell[L][b] if q == rd])
        out.append(x / y - 1)
    return " ".join(f"{v*100:+.1f}%" for v in out) or "-"
for L in sorted(cell):
    s = {a: summary.get((L, a)) for a in ARMS}
    f = lambda x, y: f"{(s[x]['ttft']/s[y]['ttft']-1)*100:+.1f}%" if s[x] and s[y] else "-"
    sep = "-"
    if s["native"] and s["transposed"]:
        sep = "yes" if max(s["native"]["t"]) < min(s["transposed"]["t"]) else "no (ranges overlap)"
    print(f"| {L} | {f('native','off')} | {f('native','transposed')} | {f('transposed','off')} | {paired(L,'native','transposed')} | {paired(L,'native','off')} | {sep} |")
