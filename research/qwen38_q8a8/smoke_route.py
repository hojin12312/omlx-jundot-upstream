import os
os.environ["OMLX_OQ_A8"] = "1"
import numpy as np, mlx.core as mx, mlx.nn as nn
from q8common import *
from omlx.patches import qwen35_oq_a8 as oa

def qlin(K, N, bits=8):
    l = nn.QuantizedLinear(K, N, bias=False, group_size=64, bits=bits)
    w = mx.random.normal((N, K)).astype(mx.bfloat16) * 0.05
    l.weight, l.scales, l.biases = mx.quantize(w, group_size=64, bits=bits)
    mx.eval(l.parameters()); return l
class MLP:
    def __init__(s): s.gate_proj=qlin(5120,3072); s.up_proj=qlin(5120,3072); s.down_proj=qlin(3072,5120)
mlp = MLP()
x = mx.random.normal((1, 512, 5120)).astype(mx.bfloat16)
act = lambda g,u: nn.silu(g)*u
ref = mlp.down_proj(act(mlp.gate_proj(x), mlp.up_proj(x)))
got = oa.oq_a8_mlp(mlp, x, act)
mx.eval(ref, got)
r=np.array(ref.astype(mx.float32)); g=np.array(got.astype(mx.float32))
print("plan", oa.classify_linear(mlp.gate_proj), "relL2", np.linalg.norm(r-g)/np.linalg.norm(r))
class G: pass
gd = G(); gd.in_proj_qkv=qlin(5120,2048); gd.in_proj_z=qlin(5120,1024); gd.in_proj_a=qlin(5120,48); gd.in_proj_b=qlin(5120,48)
outs = oa.oq_a8_gdn_projections(gd, x)
print({k:v.shape for k,v in outs.items()}) if outs else print("None")
for k,v in outs.items():
    rr=np.array(getattr(gd,k)(x).astype(mx.float32)); gg=np.array(v.astype(mx.float32)); print(k, np.linalg.norm(rr-gg)/np.linalg.norm(rr))
