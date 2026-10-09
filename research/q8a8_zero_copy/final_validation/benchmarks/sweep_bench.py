"""All 400 Q8 projections of a checkpoint, in forward order, group-major kernel vs checkpoint-layout kernel.

The earlier kernel benchmarks re-ran one weight, so the weight and its metadata stayed in cache. Here every projection
reads its own weight once per pass (about 28 GB streamed per pass), as a real prefill does. One pass = one forward
pass worth of A8 GEMMs; Stage A is hoisted per input width. Arms alternate every round.
"""
import argparse, glob, json, os, re, statistics, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ms", type=int, nargs="+", default=[512, 2048, 4096])
ap.add_argument("--rounds", type=int, default=6)
ap.add_argument("--family", choices=["all", "mlp", "rest"], default="all", help="mlp = gate/up/down (a 17408 dimension)")
ap.add_argument("--group", type=int, default=4, help="projections evaluated together (independent kernels may overlap on the GPU)")
ap.add_argument("--act-mode", type=int, default=1)
a = ap.parse_args()
sys.path.insert(0, a.tree)
import mlx.core as mx
import numpy as np
import omlx
from omlx.custom_kernels.qwen35_prefill import fast
assert str(Path(omlx.__file__).resolve()).startswith(a.tree)

PAT = re.compile(r"language_model\.model\.layers\.(\d+)\.(mlp\.(gate|up|down)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|self_attn\.[qkvo]_proj)\.weight$")
tensors = {}
for f in sorted(glob.glob(os.path.join(a.checkpoint, "*.safetensors"))):
    tensors.update(mx.load(f))
projs = []
for k in tensors:
    mt = PAT.match(k)
    if mt and tensors[k].dtype == mx.uint32:
        base = k[:-len(".weight")]
        projs.append((int(mt.group(1)), base, tensors[k], tensors[base + ".scales"], tensors[base + ".biases"]))
projs.sort(key=lambda t: (t[0], t[1]))
if a.family != "all":
    is_mlp = lambda p: 17408 in (p[2].shape[0], p[2].shape[1] * 4)
    projs = [p for p in projs if is_mlp(p) == (a.family == "mlp")]
print("projections:", len(projs), flush=True)
dtype = projs[0][3].dtype
mx.eval([p[2] for p in projs])
trans = [(mx.contiguous(p[3].T), mx.contiguous(p[4].T)) for p in projs]   # the group-major copies (1.4 GiB)
mx.eval([t for pair in trans for t in pair])
print("weights + copies resident, active GiB", mx.get_active_memory() / 2**30, flush=True)

res = {"tree": a.tree, "checkpoint": a.checkpoint, "n_proj": len(projs), "act_mode": a.act_mode, "by_m": {}}
for M in a.ms:
    stages = {}
    for p in projs:
        K = p[3].shape[1] * 64
        if K not in stages:
            x = mx.array((np.random.default_rng(K).standard_normal((M, K)) * 0.7).astype(np.float32), dtype)
            stages[K] = fast.qwen35_oq_a8_stage_a_natural(x, a.act_mode)
            mx.eval(*stages[K])

    def one_pass(native):
        t0 = time.perf_counter()
        for i0 in range(0, len(projs), a.group):         # evaluate in groups of --group projections
            outs = []
            for i in range(i0, min(i0 + a.group, len(projs))):
                _, _, w, s, b = projs[i]
                qa, sa, ra = stages[s.shape[1] * 64]
                if native:
                    outs.append(fast.qwen35_oq_a8_qmm_t(qa, sa, ra, w, s, b, 8, a.act_mode, 806, native_meta=True))
                else:
                    sg, bg = trans[i]
                    outs.append(fast.qwen35_oq_a8_qmm_t(qa, sa, ra, w, sg, bg, 8, a.act_mode, 806))
            mx.eval(outs)
        mx.synchronize()
        return (time.perf_counter() - t0) * 1e3

    for native in (False, True):
        one_pass(native)                                   # warm-up
    t = {"group_major": [], "native": []}
    for r in range(a.rounds):
        order = (False, True) if r % 2 == 0 else (True, False)
        for native in order:
            t["native" if native else "group_major"].append(one_pass(native))
    med = {k: statistics.median(v) for k, v in t.items()}
    res["by_m"][str(M)] = {"ms_per_pass": t, "median": med, "ratio_native_over_group_major": med["native"] / med["group_major"],
                           "pairwise_ratio": [n / g for n, g in zip(t["native"], t["group_major"])]}
    print(M, {k: round(v, 1) for k, v in med.items()}, "ratio", round(med["native"] / med["group_major"], 3), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))
