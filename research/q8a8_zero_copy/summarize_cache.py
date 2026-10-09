"""Prefix-cache scenario summary: per step, per arm; text equality of native vs transposed (must be identical)."""
import glob, json, sys
d = sys.argv[1]
GIB = 2**30
arms = {}
for f in sorted(glob.glob(d + "/m1_cache_*.json")):
    j = json.load(open(f)); arms[j["arm"]] = j
order = ["off", "transposed", "native", "transposed_mtp", "native_mtp"]
labels = [r["label"] for r in arms["native"]["requests"]]
print("| step | arm | prompt tok | cached | TTFT s | Q8 A8 rows per call (calls) | native/transposed calls | peak GiB | settled GiB | prepared layers | text sha |")
print("|---|---|---:|---:|---:|---|---|---:|---:|---:|---|")
for lab in labels:
    for a in order:
        if a not in arms: continue
        rs = [r for r in arms[a]["requests"] if r["label"] == lab]
        if not rs: continue
        r = rs[0]; c = r["counts"]
        print(f"| {lab} | {a} | {r['prompt_tokens']} | {r['cached_tokens']} | {r['ttft_s']:.3f} | {r['rows_hist'] or '-'} | "
              f"{c.get('a8_q8_native',0)}/{c.get('a8_q8_transposed',0)} | {r['mem_after']['peak']/GIB:.2f} | {r['mem_settled']['active']/GIB:.2f} | "
              f"{r['prepared_scan']['layers_with_prepared_copy']} | {r['text_sha256']} |")
print()
for x, y in (("native", "transposed"), ("native_mtp", "transposed_mtp")):
    if x in arms and y in arms:
        bad = [(a["label"]) for a, b in zip(arms[x]["requests"], arms[y]["requests"]) if a["text"] != b["text"] or a["cached_tokens"] != b["cached_tokens"] or a["prompt_tokens"] != b["prompt_tokens"]]
        print(f"{x} vs {y}: {len(arms[x]['requests'])} steps, text/cached/prompt mismatches: {bad or 'none'}")
for a in arms:
    ps = [r["prepared_scan"]["layers_with_prepared_copy"] for r in arms[a]["requests"]]
    print(a, "max prepared layers", max(ps), "suspect log lines:", len(arms[a].get("log_suspect_lines", [])))
