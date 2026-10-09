import glob, json, sys
d = sys.argv[1]; GIB = 2**30
print("| session | scenario | wall s | requests ok / errors | peak GiB | quiet active GiB | quiet phys GiB |")
print("|---|---|---:|---|---:|---:|---:|")
summ = []
for f in sorted(glob.glob(d + "/*.json")):
    j = json.load(open(f)); arm = j["arm"]
    for s in j["scenarios"]:
        rq = s["requests"]; err = [r for r in rq if r.get("error") and not r.get("cancelled")]
        print(f"| {arm} | {s['name']} | {s['wall']:.1f} | {len(rq)-len(err)} / {len(err)} | {s['end']['mem']['peak']/GIB:.2f} | "
              f"{s['quiet']['mem']['active']/GIB:.3f} | {s['quiet']['mem']['phys_footprint']/GIB:.3f} |")
    loop = j.get("loop", [])
    act = [l["sample"]["mem"]["active"] for l in loop]
    errs = sum(1 for l in loop if l["request"].get("error"))
    ur = j.get("unload_reload", [])
    fin = j.get("final_sample", {}).get("mem", {})
    prepared_max = 0
    summ.append((arm, len(loop), errs, min(act)/GIB if act else None, max(act)/GIB if act else None, fin.get("active", 0)/GIB, len(ur),
                 [round(u["after_unload"]["active"]/GIB, 2) for u in ur], [round(u["after_request"]["mem"]["active"]/GIB, 3) for u in ur],
                 len(j.get("log_suspect_lines", [])), (j.get("scenarios") or [{}])[0].get("end", {}).get("counters", {})))
print()
print("| session | loop requests | errors | active min..max GiB | final active GiB | unload/reload cycles | active after unload | active after reload+request | suspect log lines |")
print("|---|---:|---:|---|---:|---:|---|---|---:|")
for a in summ:
    print(f"| {a[0]} | {a[1]} | {a[2]} | {a[3]:.3f}..{a[4]:.3f} | {a[5]:.3f} | {a[6]} | {a[7]} | {a[8]} | {a[9]} |")
