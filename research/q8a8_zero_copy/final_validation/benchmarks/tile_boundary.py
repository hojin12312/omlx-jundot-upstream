"""MLP-down (N=5120, K=17408, act mode 1) around the 1024/1025 tile switch.

Arms alternate every round: group-major kernel, checkpoint-layout kernel with the (2,2) tile, and with the (1,2) tile.
The tile is forced through OMLX_TILE_TEST (22 or 12), which exists only in a scratch build and not in the PR.
Each pass streams every layer's own down_proj weight once, as a prefill does.
"""
import argparse, glob, json, os, re, statistics, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ms", type=int, nargs="+", default=[1023, 1024, 1025, 1026, 2048])
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--layers", type=int, default=16)
a = ap.parse_args()
sys.path.insert(0, a.tree)
import mlx.core as mx
import numpy as np
import omlx
from omlx.custom_kernels.qwen35_prefill import fast
assert str(Path(omlx.__file__).resolve()).startswith(a.tree)

PAT = re.compile(r"language_model\.model\.layers\.(\d+)\.mlp\.down_proj\.weight$")
tensors = {}
for f in sorted(glob.glob(os.path.join(a.checkpoint, "*.safetensors"))):
    tensors.update(mx.load(f))
projs = []
for k in tensors:
    mt = PAT.match(k)
    if mt and tensors[k].dtype == mx.uint32:
        b = k[:-len(".weight")]
        projs.append((int(mt.group(1)), tensors[k], tensors[b + ".scales"], tensors[b + ".biases"]))
projs.sort(key=lambda t: t[0])
projs = projs[: a.layers]
print("down projections:", len(projs), projs[0][1].shape, flush=True)
trans = [(mx.contiguous(p[2].T), mx.contiguous(p[3].T)) for p in projs]
mx.eval([p[1] for p in projs], [t for pair in trans for t in pair])
K = projs[0][2].shape[1] * 64
dtype = projs[0][2].dtype
ARMS = ["group_major", "native_22", "native_12"]
res = {"tree": a.tree, "n_layers": len(projs), "by_m": {}}
for M in a.ms:
    x = mx.array((np.random.default_rng(M).standard_normal((M, K)) * 0.7).astype(np.float32), dtype)
    qa, sa, ra = fast.qwen35_oq_a8_stage_a_natural(x, 1)
    mx.eval(qa, sa, ra)

    def one_pass(arm):
        os.environ["OMLX_TILE_TEST"] = "22" if arm == "native_22" else "12"
        t0 = time.perf_counter()
        for i, (_, w, s, b) in enumerate(projs):
            if arm == "group_major":
                y = fast.qwen35_oq_a8_qmm_t(qa, sa, ra, w, trans[i][0], trans[i][1], 8, 1, 806)
            else:
                y = fast.qwen35_oq_a8_qmm_t(qa, sa, ra, w, s, b, 8, 1, 806, native_meta=True)
            mx.eval(y)
        mx.synchronize()
        return (time.perf_counter() - t0) * 1e3

    for arm in ARMS:
        one_pass(arm)
    t = {arm: [] for arm in ARMS}
    for r in range(a.rounds):
        order = ARMS if r % 2 == 0 else ARMS[::-1]
        for arm in order:
            t[arm].append(one_pass(arm))
    med = {k: statistics.median(v) for k, v in t.items()}
    res["by_m"][str(M)] = {"ms_per_pass": t, "median": med,
                           "native22_over_group": med["native_22"] / med["group_major"],
                           "native12_over_group": med["native_12"] / med["group_major"],
                           "native22_over_native12": med["native_22"] / med["native_12"]}
    print(M, {k: round(v, 2) for k, v in med.items()},
          "n22/g %.3f n12/g %.3f n22/n12 %.3f" % (med["native_22"] / med["group_major"], med["native_12"] / med["group_major"], med["native_22"] / med["native_12"]), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))
