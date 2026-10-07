import json, struct, sys, glob, re, collections
snap = glob.glob('$HOME/.cache/huggingface/hub/models--Jundot--Qwen3.8-27B-oQ8e-mtp/snapshots/*')[0]
def header(p):
    with open(p,'rb') as f:
        n=struct.unpack('<Q',f.read(8))[0]; return json.loads(f.read(n))
out={}
for p in sorted(glob.glob(snap+'/model-*.safetensors')):
    try: h=header(p)
    except Exception as e: print('skip',p,e); continue
    for k,v in h.items():
        if k!='__metadata__': out[k]=(v['dtype'],v['shape'],p.split('/')[-1])
print(len(out),'tensors')
# per-layer-0 and layer-3 listing
for k,(d,s,f) in sorted(out.items()):
    if re.search(r'layers\.(0|3)\.',k) and 'language_model' in k or ('mtp' in k and re.search(r'mtp.*layers\.0\.',k)): print(k,d,s,f)
