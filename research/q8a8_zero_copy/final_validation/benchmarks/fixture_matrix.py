"""Bit-identity of the checkpoint-layout Q8 kernel against the group-major kernel over a wide synthetic matrix.

Both kernels come from one build (the prep tree), so any difference is the kernel and not the toolchain.
Shapes include K and N values the Qwen3.8-27B checkpoint does not use, small N, odd M residues, the 1024/1025 tile
switch, boundary code patterns and extreme scales. Writes a JSON summary and exits 1 on any mismatch.
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--quick", action="store_true")
a = ap.parse_args()
sys.path.insert(0, a.tree)
import mlx.core as mx
import omlx
from omlx.custom_kernels.qwen35_prefill import fast
assert str(Path(omlx.__file__).resolve()).startswith(a.tree), omlx.__file__
assert fast.oq_a8_available()

SHAPES = [(64, 64), (128, 64), (256, 192), (320, 128), (1024, 1024), (2048, 1024), (5120, 1024), (1536, 2112),
          (3072, 576), (4096, 11008), (11008, 4096), (2560, 9728), (5120, 17408), (17408, 5120), (5120, 10240),
          (6144, 5120), (5120, 6144), (5120, 12288), (5120, 1024)]
MS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 130, 255, 257, 511, 513,
      1023, 1024, 1025, 1026, 1027, 1028, 2047, 2049, 3000, 4093, 4096]
if a.quick:
    SHAPES, MS = SHAPES[:6], MS[::4]
PATTERNS = ["random", "all255", "all0", "alt127_128", "edge4"]
SCALES = ["normal", "tiny", "large"]


def make(n, k, dtype, pattern, scale, seed):
    rng = np.random.default_rng(seed)
    if pattern == "random":
        codes = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    elif pattern == "all255":
        codes = np.full((n, k), 255, np.uint8)
    elif pattern == "all0":
        codes = np.zeros((n, k), np.uint8)
    elif pattern == "alt127_128":
        codes = np.tile(np.array([127, 128], np.uint8), (n, k // 2))
    else:
        codes = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
        codes[:, :4] = [0, 127, 128, 255]
    lo, hi = {"normal": (0.001, 0.05), "tiny": (1e-4, 3e-4), "large": (0.5, 4.0)}[scale]
    w = mx.array(codes.view("<u4").copy())
    s = mx.array(rng.uniform(lo, hi, (n, k // 64)).astype(np.float32), dtype)
    b = mx.array(rng.uniform(-3.0, 1.0, (n, k // 64)).astype(np.float32) * (hi / 0.05), dtype)
    return w, s, b


def bits(x):
    return np.array(x.astype(mx.float32)).view(np.uint32)


total = bad = 0
fails = []
t0 = time.time()
for (k, n), dtype, act in itertools.product(SHAPES, (mx.bfloat16, mx.float16), (0, 1)):
    for pattern, scale in [("random", "normal"), ("all255", "normal"), ("all0", "large"), ("alt127_128", "tiny"),
                           ("edge4", "normal")]:
        if a.quick and (pattern, scale) != ("random", "normal"):
            continue
        w, s, b = make(n, k, dtype, pattern, scale, seed=k * 7 + n)
        sg, bg = mx.contiguous(s.T), mx.contiguous(b.T)
        mx.eval(w, s, b, sg, bg)
        for m in MS:
            x = mx.array((np.random.default_rng(m + k).standard_normal((m, k)) * 0.7).astype(np.float32), dtype)
            if pattern == "edge4" and m % 3 == 0:
                x = x * 40.0
            qa, sa, ra = fast.qwen35_oq_a8_stage_a_natural(x, act)
            ref = fast.qwen35_oq_a8_qmm_t(qa, sa, ra, w, sg, bg, 8, act, 806)
            got = fast.qwen35_oq_a8_qmm_t(qa, sa, ra, w, s, b, 8, act, 806, native_meta=True)
            mx.eval(ref, got)
            total += 1
            eq = bool((bits(ref) == bits(got)).all())
            if not eq:
                bad += 1
                fails.append({"k": k, "n": n, "dtype": str(dtype), "act": act, "pattern": pattern, "scale": scale, "m": m})
    print(f"{(k, n)} {dtype} act{act}: cases={total} bad={bad} t={time.time() - t0:.0f}s", flush=True)

# rejected inputs must raise, never run
rej = {}
w, s, b = make(128, 256, mx.float16, "random", "normal", 1)
qa, sa, ra = fast.qwen35_oq_a8_stage_a_natural(mx.zeros((64, 256), mx.float16), 1)
for name, call in {
    "group_major_metadata": lambda: fast.qwen35_oq_a8_qmm_t(qa, sa, ra, w, mx.contiguous(s.T), mx.contiguous(b.T), 8, 1, 806, native_meta=True),
    "variant_800": lambda: fast.qwen35_oq_a8_qmm_t(qa, sa, ra, w, s, b, 8, 1, 800, native_meta=True),
    "q4": lambda: fast.qwen35_oq_a8_qmm_t(qa, sa, ra, w, s, b, 4, 1, 806, native_meta=True),
    "packed": lambda: fast.qwen35_oq_a8_qmm_t(qa, sa, ra, w, s, b, 8, 1, 806, packed=True, native_meta=True),
    "n_not_multiple_of_64": lambda: fast.qwen35_oq_a8_qmm_t(qa, sa, ra, w[:96], s[:96], b[:96], 8, 1, 806, native_meta=True),
}.items():
    try:
        mx.eval(call())
        rej[name] = "RAN"
    except Exception as exc:  # noqa: BLE001
        rej[name] = type(exc).__name__ + ": " + str(exc)[:110]
summary = {"tree": a.tree, "cases": total, "mismatches": bad, "fails": fails[:50], "rejections": rej,
           "mlx": mx.__version__, "seconds": time.time() - t0}
Path(a.out).write_text(json.dumps(summary, indent=1))
print(json.dumps({k: summary[k] for k in ("cases", "mismatches", "rejections")}, indent=1))
sys.exit(1 if bad or "RAN" in rej.values() else 0)
