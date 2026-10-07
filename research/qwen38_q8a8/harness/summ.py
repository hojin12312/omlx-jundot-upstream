import json,sys,statistics as st
d=json.load(open(sys.argv[1]))
conds=d['conds']; base=sys.argv[2] if len(sys.argv)>2 else conds[0]
rows=[r for r in d['measurements'] if r['phase']=='timed' and r['valid']]
byL={}
for r in rows: byL.setdefault(r['length_requested'],{}).setdefault(r['sample'],{})[r['cond']]=r
for L,by in sorted(byL.items()):
    full=[g for g in by.values() if set(g)==set(conds)]
    print('L=%d  complete tuples %d'%(L,len(full)))
    for c in conds:
        print('   %-5s median pp/s %.1f  ttft %.3f'%(c, st.median(g[c]['pp_tokens_per_s'] for g in full), st.median(g[c]['ttft_s'] for g in full)))
    for c in conds:
        if c==base: continue
        ratios=[g[c]['pp_tokens_per_s']/g[base]['pp_tokens_per_s'] for g in full]
        print('   %s/%s paired: mean %+.2f%% median %+.2f%% min %+.2f%% max %+.2f%%  faster %d/%d'%(c,base,(st.mean(ratios)-1)*100,(st.median(ratios)-1)*100,(min(ratios)-1)*100,(max(ratios)-1)*100,sum(r>1 for r in ratios),len(ratios)))
