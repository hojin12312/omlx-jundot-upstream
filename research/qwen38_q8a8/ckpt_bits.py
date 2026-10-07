import json, struct, glob, re, collections, sys
snap = glob.glob('$HOME/.cache/huggingface/hub/models--Jundot--Qwen3.8-27B-oQ8e-mtp/snapshots/*')[0]
def header(p):
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n))
idx=json.load(open(snap+'/model.safetensors.index.json'))
wm=idx['weight_map']; print('tensors in index',len(wm), 'meta', idx.get('metadata'))
by_shard=collections.defaultdict(list)
for k,f in wm.items(): by_shard[f].append(k)
rows=[]; ready=[]
for f in sorted(by_shard):
    try: h=header(snap+'/'+f)
    except Exception as e: print('not ready',f); continue
    ready.append(f)
    for k in by_shard[f]:
        if k.endswith('.weight') and h[k]['dtype']=='U32':
            base=k[:-7]; sc=h.get(base+'.scales'); 
            if not sc: continue
            K=sc['shape'][1]*64; N=h[k]['shape'][0]; bits=h[k]['shape'][1]*32/K
            rows.append((base,N,K,bits,sc['dtype']))
print('shards read',ready)
cnt=collections.Counter()
for base,N,K,bits,dt in rows:
    kind=re.sub(r'language_model\.model\.layers\.\d+\.','L.',base); kind=re.sub(r'mtp\.layers\.\d+\.','mtp.L.',kind)
    cnt[(kind,N,K,bits,dt)]+=1
for k,v in sorted(cnt.items()): print(v,k)
