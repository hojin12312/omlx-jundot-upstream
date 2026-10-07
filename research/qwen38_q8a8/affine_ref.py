"""Step 2: check MLX's Q8 affine convention and the centered-INT8 algebra."""
import numpy as np, mlx.core as mx

rng = np.random.default_rng(0)
N, K, GS = 16, 256, 64
for dt in (mx.float16, mx.bfloat16):
    codes = rng.integers(0, 256, size=(N, K), dtype=np.uint8)
    # edge values in every group
    codes[:, 0:4] = [0, 127, 128, 255]
    codes[3, :] = 0; codes[4, :] = 255; codes[5, :] = 128; codes[6, :] = 127
    packed = mx.array(codes.view("<u4"))            # little-endian bytes in word
    s32 = rng.uniform(0.001, 0.05, size=(N, K // GS)).astype(np.float32)
    b32 = rng.uniform(-3, 3, size=(N, K // GS)).astype(np.float32)
    s = mx.array(s32).astype(dt); b = mx.array(b32).astype(dt)
    deq = mx.dequantize(packed, s, b, group_size=GS, bits=8, mode="affine").astype(mx.float32)
    sf = np.array(s.astype(mx.float32)); bf = np.array(b.astype(mx.float32))
    # w = s*q_u8 + b   (float64 reference)
    ref_u = sf.repeat(GS, 1).astype(np.float64) * codes + bf.repeat(GS, 1).astype(np.float64)
    # centered: q_s = q_u8 - 128 (XOR 0x80 as int8), w = s*q_s + (b + 128 s)
    qs = (codes ^ 0x80).view(np.int8).astype(np.int64)
    assert np.array_equal(qs, codes.astype(np.int64) - 128)
    ref_c = sf.repeat(GS, 1).astype(np.float64) * qs + (bf + 128 * sf).repeat(GS, 1).astype(np.float64)
    d = np.array(deq)
    print(dt, "max|deq-ref_u|", np.abs(d - ref_u).max(), " max|ref_u-ref_c|", np.abs(ref_u - ref_c).max(),
          " max|w|", np.abs(ref_u).max())
    # A8 dot product identity per group: sum a*w = s*acc_s + (b+128 s)*r
    a = rng.integers(-127, 128, size=(8, K)).astype(np.int64)
    exact = (a.astype(np.float64) @ ref_u.T)
    acc = np.zeros((8, N)); 
    for g in range(K // GS):
        sl = slice(g * GS, (g + 1) * GS)
        accg = a[:, sl] @ qs[:, sl].T                 # int
        rg = a[:, sl].sum(1, keepdims=True)           # Ra
        acc += sf[:, g] * accg + (bf[:, g] + 128 * sf[:, g]) * rg
    print("   centered-int8 dot vs exact: max rel", np.abs(acc - exact).max() / np.abs(exact).max())
