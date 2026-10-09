"""Q4/Q5 (and Q8 group-major) W8A8 outputs must be bit-identical between two trees: run once per tree, then compare.

usage: q45_unchanged.py TREE OUT.npz
"""
import sys
from pathlib import Path
import numpy as np

tree, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, tree)
import mlx.core as mx
import omlx
from omlx.custom_kernels.qwen35_prefill import fast

assert str(Path(omlx.__file__).resolve()).startswith(tree)
res = {}
for bits, variant in ((4, 806), (4, 800), (5, 800), (8, 806), (8, 800)):
    for dtype in (mx.float16, mx.bfloat16):
        for act in (0, 1):
            for (k, n, m) in ((512, 256, 130), (2048, 1024, 2048), (1024, 128, 1025)):
                rng = np.random.default_rng(bits * 1000 + k + m)
                w = mx.array(rng.standard_normal((n, k)).astype(np.float32) * 0.05, dtype)
                packed, s, b = mx.quantize(w, group_size=64, bits=bits, mode="affine")
                x = mx.array((rng.standard_normal((m, k)) * 0.5).astype(np.float32), dtype)
                try:
                    qa, sa, ra = fast.qwen35_oq_a8_stage_a_natural(x, act)
                    y = fast.qwen35_oq_a8_qmm_t(qa, sa, ra, packed, mx.contiguous(s.T), mx.contiguous(b.T), bits, act, variant)
                    mx.eval(y)
                    res[f"b{bits}_v{variant}_{dtype}_a{act}_{k}x{n}x{m}"] = np.array(y.astype(mx.float32)).view(np.uint32)
                except Exception as exc:  # noqa: BLE001
                    res[f"b{bits}_v{variant}_{dtype}_a{act}_{k}x{n}x{m}_ERR"] = np.frombuffer(str(exc)[:80].encode().ljust(80, b" "), np.uint8)
np.savez(out, **res)
print(tree, len(res), "entries,", sum(1 for k in res if k.endswith("_ERR")), "errors")
