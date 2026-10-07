"""Save Q4/Q5 A8 outputs on fixed inputs; run once per tree and compare the npz files bit for bit.

usage: q45_identity.py <tree> <out.npz>
"""
import sys, os
tree, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree)
import numpy as np, mlx.core as mx
import omlx
assert omlx.__file__.startswith(tree + "/"), omlx.__file__
from omlx.custom_kernels.qwen35_prefill import fast
assert fast.oq_a8_available()
res = {}
rng = np.random.default_rng(2026)
for bits in (4, 5):
    for dt, dn in ((mx.bfloat16, "bf16"), (mx.float16, "fp16")):
        for (M, N, K) in ((1, 128, 64), (33, 256, 320), (257, 192, 512), (2048, 128, 1024)):
            w = mx.array((rng.standard_normal((N, K)) * 0.05).astype(np.float32)).astype(dt)
            packed, sc, bi = mx.quantize(w, group_size=64, bits=bits, mode="affine")
            x = mx.array(rng.standard_normal((M, K)).astype(np.float32)).astype(dt)
            sg, bg = mx.contiguous(sc.T), mx.contiguous(bi.T)
            for am in (0, 1):
                qa, sa, ra = fast.qwen35_oq_a8_stage_a_v8(x, am)
                for v in (800, 801, 802, 803, 804, 805, 806):
                    if N % {800: 64, 801: 64, 802: 128, 803: 128, 804: 128, 805: 64, 806: 64}[v]:
                        continue
                    y = fast.qwen35_oq_a8_qmm_t(qa, sa, ra, packed, sg, bg, bits, am, v)
                    mx.eval(y)
                    res[f"q{bits}_{dn}_M{M}_N{N}_K{K}_am{am}_v{v}"] = np.array(y.astype(mx.float32))
                if bits == 5 or True:
                    dec = fast.qwen35_oq_a8_decode_weights(packed, bits, K // 64); mx.eval(dec)
                    res[f"dec_q{bits}_{dn}_N{N}_K{K}"] = np.array(dec)
np.savez(out, **res)
print(len(res), "arrays ->", out)
