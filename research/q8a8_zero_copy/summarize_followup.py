"""Upstream main (merged #4350) against the follow-up commit, same prompts, two rounds in opposite order (MTP off)."""
import json, glob, statistics as st, collections, sys
GIB = 2**30; d = sys.argv[1]
rec = collections.defaultdict(lambda: collections.defaultdict(list)); pairs = collections.defaultdict(lambda: collections.defaultdict(dict))
for f in sorted(glob.glob(d + "/fu_*_mtpoff.json")):
    j = json.load(open(f)); arm = j["tag"].split("_r")[0].removeprefix("fu_")
    print(f"- session {arm} round {j['round']}: tree {j['tree_git_head'][:8]}, {len(j['requests'])} requests")
    for r in j["requests"]:
        if r["role"] == "primary" and r["client_ttft_s"] and r["prompt_tokens"] == r["length"] and not r["cached_tokens"]:
            rec[r["length"]][arm].append(r); pairs[r["length"]][r["nonce"]][arm] = r["prompt_tokens"] / r["client_ttft_s"]
print("\n| L | n | main tok/s | follow-up tok/s | paired gain median (min..max) | TTFT main -> follow-up (s) | peak main / follow-up (GiB) | settled active main / follow-up (GiB) | phys main / follow-up (GiB) |")
print("|---:|---:|---:|---:|---|---|---|---|---|")
md = st.median
for L in sorted(rec):
    m, f = rec[L]["main"], rec[L]["followup"]
    g = [v["followup"] / v["main"] - 1 for v in pairs[L].values() if "main" in v and "followup" in v]
    print(f"| {L} | {len(f)} | {md([r['prompt_tokens']/r['client_ttft_s'] for r in m]):.0f} | {md([r['prompt_tokens']/r['client_ttft_s'] for r in f]):.0f} | "
          f"{md(g)*100:+.1f}% ({min(g)*100:+.1f}..{max(g)*100:+.1f}) | {md([r['client_ttft_s'] for r in m]):.3f} -> {md([r['client_ttft_s'] for r in f]):.3f} | "
          f"{md([r['mem_after']['peak'] for r in m])/GIB:.2f} / {md([r['mem_after']['peak'] for r in f])/GIB:.2f} | "
          f"{md([r['mem_settled_15s']['active'] for r in m])/GIB:.3f} / {md([r['mem_settled_15s']['active'] for r in f])/GIB:.3f} | "
          f"{md([r['mem_settled_15s']['phys_footprint'] for r in m])/GIB:.2f} / {md([r['mem_settled_15s']['phys_footprint'] for r in f])/GIB:.2f} |")
print()
for arm in ("main", "followup"):
    for rd in (1, 2):
        j = json.load(open(f"{d}/fu_{arm}_r{rd}_mtpoff.json")); w = [r for r in j["requests"] if r["role"] == "warmup"][0]; c = w["counters_1s"]
        print(f"- {arm} round {rd}, first A8 request (L={w['length']}): TTFT {w['client_ttft_s']:.3f} s, prepare_builds {c.get('prepare_builds', 0)}, "
              f"native calls {c.get('a8_q8_native', 0)}, group-major calls {c.get('a8_q8_transposed', 0)}, active delta {w['mem_settled_15s']['active'] - w['mem_before']['active']:,} B")
t = collections.defaultdict(dict)
for f in glob.glob(d + "/fu_*_mtpoff.json"):
    j = json.load(open(f)); arm = j["tag"].split("_r")[0].removeprefix("fu_")
    for r in j["requests"]: t[(j["round"], r["length"], r["nonce"])][arm] = r["text"]
both = [v for v in t.values() if len(v) == 2]
print(f"\nOutput text, same prompt: {len(both)} pairs, {sum(v['main'] == v['followup'] for v in both)} identical")
