"""In-process whole-model prefill chunk, group-major vs checkpoint-layout Q8 kernel, alternating inside one process.

Same weights, same process, same activations: only the per-projection plan changes between arms. Isolates the A8 route
from server, scheduler and process-level effects. ``--subset`` limits the switch to one projection family so a
residual gap can be located (mlp / linear_attn / self_attn / all).
"""
import argparse, json, statistics, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tree", required=True)
ap.add_argument("--checkpoint", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rows", type=int, nargs="+", default=[512, 2048])
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--wired", action="store_true", help="mx.set_wired_limit(recommended working set) before loading, as the server does")
ap.add_argument("--subsets", nargs="+", default=["all"])
a = ap.parse_args()
sys.path.insert(0, a.tree)
import mlx.core as mx
import numpy as np
import omlx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from omlx.patches import qwen35_oq_a8 as d
assert str(Path(omlx.__file__).resolve()).startswith(a.tree)

if a.wired:
    rec = mx.device_info()["max_recommended_working_set_size"]
    print("wired limit set to", rec / 2**30, "GiB (previous", mx.set_wired_limit(rec) / 2**30, ")", flush=True)
model, tok = load(a.checkpoint)
d.apply_qwen35_oq_a8_patch(model, min_tokens=128)
mods = [(n, m) for n, m in model.named_modules()]
print("modules", len(mods), flush=True)


def set_arm(native, subset):
    """Re-plan every module: members of ``subset`` follow ``native``; the rest stay native (the default)."""
    import os
    for name, m in mods:
        if hasattr(m, d._PLAN_ATTR):
            object.__delattr__(m, d._PLAN_ATTR)
    want_transposed = {}
    orig = d._native_meta_for
    state = {"cur": None}

    for name, m in mods:
        want = native or not (subset == "all" or f".{subset}." in f".{name}.")
        os.environ["OMLX_OQ_A8_Q8_NATIVE_META"] = "1" if want else "0"
        if isinstance(m, d.nn.QuantizedLinear):
            d.classify_linear(m)
    os.environ.pop("OMLX_OQ_A8_Q8_NATIVE_META", None)


def n_native():
    return sum(1 for _, m in mods if getattr(m, d._PLAN_ATTR, None) is not None and getattr(m, d._PLAN_ATTR).native_meta)


rng = np.random.default_rng(0)
res = {"tree": a.tree, "rows": {}}
for rows in a.rows:
    ids = mx.array(rng.integers(1000, 200000, size=(1, rows)).astype(np.int32))

    def fwd():
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        out = model(ids, cache=cache)
        mx.eval(out)
        mx.synchronize()
        return (time.perf_counter() - t0) * 1e3

    for subset in a.subsets:
        t = {"transposed": [], "native": []}
        for native in (False, True, False, True):          # warm-up of both arms; builds the group-major copies
            set_arm(native, subset)
            fwd()
        for r in range(a.rounds):
            for native in ((False, True) if r % 2 == 0 else (True, False)):
                set_arm(native, subset)
                t["native" if native else "transposed"].append(fwd())
        med = {k: statistics.median(v) for k, v in t.items()}
        res["rows"][f"{rows}.{subset}"] = {"ms": t, "median": med, "ratio": med["native"] / med["transposed"],
                                           "pairwise": [n / g for n, g in zip(t["native"], t["transposed"])]}
        print(rows, subset, {k: round(v, 1) for k, v in med.items()}, "ratio", round(med["native"] / med["transposed"], 3), flush=True)
Path(a.out).write_text(json.dumps(res, indent=1))
