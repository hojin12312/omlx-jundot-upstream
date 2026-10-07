"""Phase 1 microbench: stock MLX Q8 / existing oMLX Q8-A16 native / Q8A8 (same weight, same x)."""
import argparse, json, statistics, sys, time
import numpy as np, mlx.core as mx
from q8common import *

ap = argparse.ArgumentParser()
ap.add_argument("--shapes", default="gate,down,qkv,z,out,q,k")
ap.add_argument("--ms", default="32,64,128,256,512,1024,2048,4096,8192")
ap.add_argument("--variants", default="806")
ap.add_argument("--layouts", default="1")
ap.add_argument("--act-mode", type=int, default=0)
ap.add_argument("--reps", type=int, default=6)
ap.add_argument("--samples", type=int, default=11)
ap.add_argument("--out", default=None)
ap.add_argument("--no-a16", action="store_true")
a = ap.parse_args()

SRC = {  # name -> (prefix, K, N)
    "gate": ("language_model.model.layers.0.mlp.gate_proj", 5120, 17408),
    "down": ("language_model.model.layers.0.mlp.down_proj", 17408, 5120),
    "qkv": ("language_model.model.layers.0.linear_attn.in_proj_qkv", 5120, 10240),
    "z": ("language_model.model.layers.0.linear_attn.in_proj_z", 5120, 6144),
    "out": ("language_model.model.layers.0.linear_attn.out_proj", 6144, 5120),
    "q": ("language_model.model.layers.3.self_attn.q_proj", 5120, 12288),
    "k": ("language_model.model.layers.3.self_attn.k_proj", 5120, 1024),
}
variants = [int(v) for v in a.variants.split(",")]
layouts = [int(v) for v in a.layouts.split(",")]
Ms = [int(v) for v in a.ms.split(",")]
mx.set_wired_limit(8 * 2**30)  # model is only 27 GiB but benchmarks must not grow unbounded
d = mx.load(os.path.join(SNAP, "model-00001-of-00006.safetensors"))
rng = np.random.default_rng(3)
fout = open(a.out, "a") if a.out else None

def timeit(fn, reps):
    outs = [fn() for _ in range(reps)]
    t0 = time.perf_counter(); mx.eval(outs); mx.synchronize()
    return (time.perf_counter() - t0) / reps * 1e3

for name in a.shapes.split(","):
    prefix, K, N = SRC[name]
    w, s, b = d[prefix + ".weight"], d[prefix + ".scales"], d[prefix + ".biases"]
    sg, bg = group_major(s, b)
    mx.eval(w, s, b, sg, bg)
    assert w.shape == (N, K // 4), (w.shape, N, K)
    for M in Ms:
        x = mx.array(rng.standard_normal((M, K)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(x)
        conds = {}
        conds["stock"] = lambda: mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=8)
        if not a.no_a16:
            conds["a16"] = lambda: fast.qwen35_q8_affine_qmm_t(x, w, s, b, 8, 64)
        for lay in layouts:
            st = stage_a(x, a.act_mode, lay); mx.eval(*st)
            for v in variants:
                if N % TILES[v][1]: continue
                ms_, mb_ = (s, b) if lay == 2 else (sg, bg)
                conds[f"gemm_l{lay}_v{v}"] = (lambda st=st, lay=lay, v=v, ms_=ms_, mb_=mb_: q8a8(x, w, ms_, mb_, a.act_mode, v, lay, stage=st))
                conds[f"full_l{lay}_v{v}"] = (lambda lay=lay, v=v, ms_=ms_, mb_=mb_: q8a8(x, w, ms_, mb_, a.act_mode, v, lay))
            conds[f"stageA_l{lay}"] = (lambda lay=lay: stage_a(x, a.act_mode, lay)[0])
        names = list(conds)
        reps = max(2, min(a.reps, int(2e11 / (2 * M * N * K)) + 1)) if M * N * K < 2e10 else a.reps
        for n in names: timeit(conds[n], 2)            # warmup (compile + first use)
        samples = {n: [] for n in names}
        for i in range(a.samples):
            order = names[i % len(names):] + names[:i % len(names)]   # rotate start
            for n in order: samples[n].append(timeit(conds[n], a.reps))
        row = dict(shape=name, K=K, N=N, M=M, reps=a.reps, samples=a.samples,
                   ms={n: statistics.median(v) for n, v in samples.items()},
                   iqr={n: float(np.subtract(*np.percentile(v, [75, 25]))) for n, v in samples.items()})
        line = json.dumps(row)
        print(line, flush=True)
        if fout: fout.write(line + "\n"); fout.flush()
        del x
