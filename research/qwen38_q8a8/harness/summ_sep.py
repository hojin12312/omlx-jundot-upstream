import json,glob,sys,statistics as st,re
tag=sys.argv[1]; runs=sys.argv[2] if len(sys.argv)>2 else '$Q8A8_ROOT/runs'
files=sorted([p for p in glob.glob(f'{runs}/{tag}_*_*.json') if re.search(rf'{tag}_\d+_(main|src)\.json$',p)], key=lambda p:int(re.search(rf'{tag}_(\d+)_',p).group(1)))
data={}
for p in files:
    m=re.search(rf'{tag}_(\d+)_(\w+)\.json',p); i,w=int(m.group(1)),m.group(2)
    d=json.load(open(p))
    data[i]=(w,{(r['length_requested'],r['sample']):r for r in d['measurements'] if r['phase']=='timed' and r['valid']})
Ls=sorted({k[0] for _,v in data.values() for k in v})
for L in Ls:
    per={'main':[], 'src':[]}
    for i,(w,v) in sorted(data.items()):
        vals=[r['pp_tokens_per_s'] for (l,s),r in sorted(v.items()) if l==L]
        per[w].append((i,vals))
        print(f'L={L} run{i} {w:<4}', ' '.join('%.1f'%x for x in vals), ' median %.1f'%st.median(vals))
    mm=[st.median(v) for _,v in per['main']]; ss=[st.median(v) for _,v in per['src']]
    print(f'L={L} main median-of-runs {st.median(mm):.1f}  src {st.median(ss):.1f}  ratio {st.median(ss)/st.median(mm):.4f}')
    # adjacent ABBA pairs: (1,2)(4,3)... use order-balanced mean of ratios
    ratios=[]
    idx=sorted(data)
    for a in range(0,len(idx)-1,2):
        i,j=idx[a],idx[a+1]
        (wi,vi),(wj,vj)=data[i],data[j]
        if {wi,wj}!={'main','src'}: continue
        mi=st.median([r['pp_tokens_per_s'] for (l,s),r in vi.items() if l==L]); mj=st.median([r['pp_tokens_per_s'] for (l,s),r in vj.items() if l==L])
        ratios.append((mj/mi if wj=='src' else mi/mj))
    print(f'L={L} adjacent-pair src/main ratios', ['%.3f'%r for r in ratios], 'mean %.4f'%(sum(ratios)/len(ratios)))
