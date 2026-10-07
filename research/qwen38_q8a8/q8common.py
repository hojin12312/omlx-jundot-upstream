import glob, os, sys
import numpy as np
import mlx.core as mx
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
import omlx
assert "q8a8/src" in omlx.__file__, omlx.__file__
from omlx.custom_kernels.qwen35_prefill import fast

SNAP = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/models--Jundot--Qwen3.8-27B-oQ8e-mtp/snapshots/*"))
SNAP = SNAP[0] if SNAP else None
TILES = {800: (64, 64), 801: (128, 64), 802: (64, 128), 803: (128, 128), 804: (32, 128),
         805: (256, 64), 806: (32, 64)}

def to_np_f64(a):
    return np.array(a.astype(mx.float32)).astype(np.float64)

def unpack_codes(weight):
    """uint32 [N, K/4] -> uint8 codes [N, K] (little-endian bytes)."""
    return np.array(weight).view(np.uint8)

def dequant64(weight, scales, biases, gs=64):
    q = unpack_codes(weight).astype(np.float64)
    s = to_np_f64(scales).repeat(gs, 1); b = to_np_f64(biases).repeat(gs, 1)
    return s * q + b

def stage_a(x, act_mode, layout=1):
    if layout in (1, 2):
        return fast.qwen35_oq_a8_stage_a_natural(x, act_mode)
    return fast.qwen35_oq_a8_stage_a_v8(x, act_mode)

def q8a8(x, weight, st_scales, st_biases, act_mode=0, variant=800, layout=1, stage=None):
    staged = layout == 2
    """x [M,K]; weight packed [N,K/4]; scales/biases group-major [G,N] contiguous."""
    qa, sa, ra = stage if stage is not None else stage_a(x, act_mode, layout)
    return fast.qwen35_oq_a8_qmm_t(qa, sa, ra, weight, st_scales, st_biases, 8,
                                   act_mode, variant, **({"staged": True} if staged else {}))

def group_major(scales, biases):
    return mx.contiguous(scales.T), mx.contiguous(biases.T)

def load_layer_tensors(prefix, shard="model-00001-of-00006.safetensors"):
    d = mx.load(os.path.join(SNAP, shard))
    return {k.split(prefix + ".")[-1]: v for k, v in d.items() if k.startswith(prefix + ".")}
