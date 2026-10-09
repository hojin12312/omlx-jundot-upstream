import json, sys, collections
d = sys.argv[1]; GIB = 2**30
A = {a: json.load(open(f"{d}/m1_chunks_{a}.json")) for a in ("off", "transposed", "native")}
labels = [r["label"] for r in A["native"]["requests"]]
print("| prompt | chunk | arm | TTFT s | prefill tok/s | Q8 A8 rows per call | native/transposed calls | peak GiB | settled GiB | text sha |")
print("|---|---|---|---:|---:|---|---|---:|---:|---|")
for lab in labels:
    for a in ("off", "transposed", "native"):
        rs = [r for r in A[a]["requests"] if r["label"] == lab]
        if not rs: continue
        r = rs[0]; c = r["counts"]; p, ch = lab.split(".")
        print(f"| {p} | {ch} | {a} | {r['ttft_s']:.3f} | {r['prompt_tokens']/r['ttft_s']:.0f} | {r['rows_hist'] or '-'} | {c.get('a8_q8_native',0)}/{c.get('a8_q8_transposed',0)} | "
              f"{r['mem_after']['peak']/GIB:.2f} | {r['mem_settled']['active']/GIB:.2f} | {r['text_sha256']} |")
print()
bad = [a["label"] for a, b in zip(A["native"]["requests"], A["transposed"]["requests"]) if a["text"] != b["text"] or a["rows_hist"] != b["rows_hist"]]
print("native vs transposed: text or row-histogram mismatches:", bad or "none", f"({len(labels)} steps)")
for a in A:
    by = collections.defaultdict(set)
    for r in A[a]["requests"]:
        by[r["label"].split(".")[0]].add(r["text_sha256"])
    print(a, "distinct output texts per prompt across chunk sizes:", {k: len(v) for k, v in by.items()})
print("max prepared layers:", {a: max(r["prepared_scan"]["layers_with_prepared_copy"] for r in A[a]["requests"]) for a in A})
