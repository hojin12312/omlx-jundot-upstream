"""Phase 1 correctness: Q8 decoder / affine algebra vs activation-quantization error."""
import itertools, json, sys
import numpy as np, mlx.core as mx
from q8common import *

def err(a, ref):
    a = to_np_f64(a) if not isinstance(a, np.ndarray) else a
    d = np.abs(a - ref)
    return dict(max_abs=float(d.max()), mean_abs=float(d.mean()),
                rel_l2=float(np.linalg.norm(a - ref) / max(np.linalg.norm(ref), 1e-30)))

def make_case(rng, M, N, K, dt, edge=True):
    codes = rng.integers(0, 256, size=(N, K), dtype=np.uint8)
    if edge:
        codes[:, 0:8] = np.array([0, 127, 128, 255, 1, 254, 129, 126], dtype=np.uint8)
        codes[0, :] = 0; codes[1, :] = 255; codes[2, :] = 128; codes[3, :] = 127
    w = mx.array(codes.view("<u4").copy())
    s = mx.array(rng.uniform(0.001, 0.03, size=(N, K // 64)).astype(np.float32)).astype(dt)
    b = mx.array(rng.uniform(-2, 2, size=(N, K // 64)).astype(np.float32)).astype(dt)
    x = mx.array(rng.standard_normal((M, K)).astype(np.float32)).astype(dt)
    return x, w, s, b

rows = []
ok = True
rng = np.random.default_rng(7)
for dt, dn in ((mx.bfloat16, "bf16"), (mx.float16, "fp16")):
    for (M, N, K) in ((1, 128, 64), (17, 128, 320), (33, 256, 512), (100, 128, 1024), (257, 256, 576)):
        x, w, s, b = make_case(rng, M, N, K, dt)
        sg, bg = group_major(s, b)
        W64 = dequant64(w, s, b)
        x64 = to_np_f64(x)
        y_fp = x64 @ W64.T                                   # bf16-activation reference (float64)
        y_stock = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=8)
        y_a16 = fast.qwen35_q8_affine_qmm_t(x, w, s, b, 8, 64) if M >= 1 else None
        for act_mode, layout, variant in itertools.product((0, 1), (1,), (800, 801, 802, 803, 804, 805, 806)):
            if N % TILES[variant][1]:
                continue
            qa, sa, ra = stage_a(x, act_mode, layout)
            y = q8a8(x, w, sg, bg, act_mode, variant, layout, stage=(qa, sa, ra))
            mx.eval(y)
            # Stage-A-quantized reference: removes activation quantization from the comparison.
            qa_n, sa_n, _ = fast.qwen35_oq_a8_quantize(x, act_mode)
            qn = np.array(qa_n).astype(np.float64)
            if act_mode == 0:
                xq = qn * np.array(sa_n).astype(np.float64)[:, None]
            else:
                xq = qn * np.array(sa_n).astype(np.float64).repeat(64, 1)
            y_q = xq @ W64.T
            # Output is rounded to dt: compare against the reference rounded the same way.
            y_q_rt = to_np_f64(mx.array(y_q.astype(np.float32)).astype(dt))
            yn = to_np_f64(y)
            r = dict(dt=dn, M=M, N=N, K=K, act_mode=act_mode, layout=layout, variant=variant,
                     finite=bool(np.isfinite(yn).all()),
                     decoder=err(yn, y_q),                    # decoder + affine + fp32 accum only
                     decoder_exact_after_round=float((yn == y_q_rt).mean()),
                     act_quant=err(yn, y_fp),                 # + activation quantization error
                     stock_vs_fp=err(y_stock, y_fp),
                     a16_vs_fp=err(y_a16, y_fp))
            rows.append(r)
            # decoder error must sit at output rounding level (<= 1 ulp of dt) or fp32 noise
            tol = 2.0 ** -7 if dn == "bf16" else 2.0 ** -10
            if not r["finite"] or r["decoder"]["rel_l2"] > tol:
                ok = False; print("FAIL", r)
json.dump(rows, open("correct_synth.json", "w"))
print("cases", len(rows), "ALL_OK" if ok else "SOME_FAIL")
for dn in ("bf16", "fp16"):
    sel = [r for r in rows if r["dt"] == dn]
    for k in ("decoder", "act_quant", "stock_vs_fp", "a16_vs_fp"):
        print(dn, k, "worst rel_l2 %.3e" % max(r[k]["rel_l2"] for r in sel),
              "median %.3e" % np.median([r[k]["rel_l2"] for r in sel]))
    print(dn, "min frac exact after rounding", min(r["decoder_exact_after_round"] for r in sel))
