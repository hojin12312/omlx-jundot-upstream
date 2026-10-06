# SPDX-License-Identifier: Apache-2.0
"""Routed Q4A8 gate/up gather kernel (``m5_gather_qmm_a8``)."""

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches import m5_gather_qmm_a8 as a8
from omlx.patches import m5_gather_qmm_nax as nax

GROUP = 64


def _a8_ready() -> bool:
    try:
        from omlx.custom_kernels.qwen35_prefill import fast

        return (
            bool(fast.oq_a8_available())
            and nax.enabled()
            and a8._get_kernel() is not None
        )
    except Exception:
        return False


needs_a8 = pytest.mark.skipif(
    not _a8_ready(), reason="needs the M5 NAX + oQ A8 kernels"
)


def _routes(kind, tokens, topk, experts, seed):
    """Sorted routes ``(idx, row_map)``; ``kind`` shapes the expert histogram."""
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        flat = rng.integers(0, experts, size=tokens * topk)
    elif kind == "skewed":
        p = rng.pareto(1.2, size=experts) + 0.02
        flat = rng.choice(experts, size=tokens * topk, p=p / p.sum())
    elif kind == "one_hot":
        flat = np.full(tokens * topk, experts // 2)
    elif kind == "empty_experts":
        flat = rng.choice(np.array([0, 1, experts - 1]), size=tokens * topk)
    elif kind == "partial_tiles":
        # every expert's row count is odd-sized, so no tile height divides it
        flat = np.concatenate([np.full(7 + 13 * (e % 5), e) for e in range(experts)])
        rng.shuffle(flat)
        tokens = len(flat) // topk
        flat = flat[: tokens * topk]
    else:
        raise ValueError(kind)
    order = np.argsort(flat, kind="stable")
    return (
        tokens,
        mx.array(flat[order].astype(np.uint32)),
        mx.array((order // topk).astype(np.uint32)),
    )


def _problem(E, N, K, dtype, kind="skewed", tokens=100, topk=10, seed=7):
    tokens, idx, rmap = _routes(kind, tokens, topk, E, seed)
    w = (mx.random.normal((E, N, K), key=mx.random.key(seed)) * 0.05).astype(dtype)
    wq, s, b = mx.quantize(w, group_size=GROUP, bits=4, mode="affine")
    x = (mx.random.normal((tokens, 1, K), key=mx.random.key(seed + 1)) * 0.6).astype(
        dtype
    )
    mx.eval(wq, s, b, x, idx, rmap)
    return x, wq, s, b, idx, rmap


def _run(problem, **kw):
    x, wq, s, b, idx, rmap = problem
    return a8.sorted_gather_qmm_a8(
        x, wq, s, b, idx, rmap, group_size=GROUP, bits=4, **kw
    )


def _contract_reference(problem):
    """The A8 contract in float32, independent of the kernel's addressing:
    ``Sa[m] * (Qa[m] @ dequantize(W)[e].T)`` with Stage A's own ``Qa``/``Sa``.
    Equal to ``Sa * sum_g (Sw_g * acc_g + Bw_g * Ra_g)`` up to float32 order."""
    x, wq, s, b, idx, rmap = problem
    from omlx.custom_kernels.qwen35_prefill import fast

    # Plain Stage A: the kernel's ``_v8`` variant permutes the K slots of every
    # group into the order its fragment loads expect (the sum is unchanged).
    qa, sa, _ = fast.qwen35_oq_a8_quantize(x, 0)
    qa = qa.reshape(qa.shape[0], -1)  # [T, 1, K] -> [T, K]
    wd = mx.dequantize(
        wq, s.astype(mx.float32), b.astype(mx.float32), group_size=GROUP, bits=4
    )
    qa_f = qa.astype(mx.float32)
    idx_np = np.array(idx)
    out = np.zeros((idx_np.shape[0], wd.shape[1]), np.float32)
    for e in np.unique(idx_np):
        rows = np.nonzero(idx_np == e)[0]
        tok = rmap[mx.array(rows)]
        y = (qa_f[tok] @ wd[int(e)].T) * sa[tok][:, None]
        out[rows] = np.array(y)
    return out


def _assert_close_to_contract(out, ref):
    got = np.array(out.astype(mx.float32)).reshape(ref.shape)
    atol = 2e-3 * float(np.abs(ref).max())
    bad = np.abs(got - ref) > 1e-2 * np.abs(ref) + atol
    assert not bad.any(), f"{bad.sum()} of {bad.size} elements off the contract"


@needs_a8
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "K",
    [
        192,  # G = 3: not a multiple of 4, scalar loads
        256,  # G = 4: staged
        320,  # G = 5: scalar loads
        2560,  # G = 40: Qwen3.8-Flash-Next, staged, 10 KiB per tile
        4352,  # G = 68: above the staging limit, scalar loads
    ],
)
def test_checkpoint_layout_metadata_matches_the_contract(K, dtype):
    """The kernel reads the checkpoint's ``[E, N, G]`` scale/bias as stored, for
    the staged loader and for both scalar fallbacks."""
    problem = _problem(6, 128, K, dtype, tokens=60)
    out = _run(problem, bm=32)
    assert out is not None and out.shape == (problem[4].shape[0], 1, 128)
    _assert_close_to_contract(out, _contract_reference(problem))


@needs_a8
def test_real_flash_next_shape_matches_the_contract():
    """K = 2560 (G = 40), fused gate+up N = 1280: the shape of the model."""
    problem = _problem(16, 1280, 2560, mx.bfloat16, tokens=300)
    out = _run(problem)
    assert out is not None and out.shape == (problem[4].shape[0], 1, 1280)
    _assert_close_to_contract(out, _contract_reference(problem))


@needs_a8
@pytest.mark.parametrize("K", [256, 2560])
def test_staged_and_scalar_loaders_are_bitwise_equal(K, monkeypatch):
    problem = _problem(8, 256, K, mx.bfloat16, tokens=80)
    assert a8._staged(K // GROUP)
    staged = _run(problem, bm=32)
    monkeypatch.setattr(a8, "_staged", lambda groups: False)
    scalar = _run(problem, bm=32)
    assert nax._bits_equal(staged, scalar)


@needs_a8
@pytest.mark.parametrize("kind", ["uniform", "skewed", "one_hot", "empty_experts", "partial_tiles"])
@pytest.mark.parametrize("bm", [32, 64])
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
def test_fused_swiglu_is_bitwise_the_unfused_chain(kind, bm, dtype):
    """Skewed, one-hot, empty-expert and partial-tile routes: the SwiGLU in the
    epilogue equals the plain A8 output run through the reference activation,
    and every output element is written."""
    problem = _problem(24, 256, 256, dtype, kind=kind, tokens=100)
    fused = _run(problem, bm=bm, swiglu=True)
    plain = _run(problem, bm=bm)
    gate, up = mx.split(plain, 2, axis=-1)
    assert nax._bits_equal(fused, nax.reference_activation(up, gate))
    filled = _run(problem, bm=bm, swiglu=True, init_value=float("nan"))
    assert nax._bits_equal(filled, fused)


@needs_a8
def test_more_than_32768_routed_rows():
    problem = _problem(16, 128, 256, mx.bfloat16, kind="uniform", tokens=3500)
    assert problem[4].shape[0] > 32768
    out = _run(problem, bm=32)
    assert out is not None
    _assert_close_to_contract(out, _contract_reference(problem))


@needs_a8
def test_kernel_reads_the_checkpoint_tensors_without_a_copy(monkeypatch):
    """The kernel receives the very scale/bias arrays of the checkpoint."""
    problem = _problem(8, 128, 256, mx.bfloat16)
    _, _, s, b, _, _ = problem
    real = a8._get_kernel()
    seen = []

    def spy(*, inputs, **kwargs):
        seen.append([id(a) for a in inputs])
        return real(inputs=inputs, **kwargs)

    monkeypatch.setattr(a8, "_get_kernel", lambda: spy)
    assert _run(problem) is not None
    # inputs: qa, sa, ra, w, scales, biases, ...
    assert seen[0][4:6] == [id(s), id(b)]


@needs_a8
def test_failed_canary_declines_to_a16(monkeypatch):
    problem = _problem(8, 128, 256, mx.bfloat16)
    monkeypatch.setitem(nax._verified, ("a8", mx.bfloat16, 256 // GROUP), False)
    assert _run(problem) is None


def _operands(bits=4, group=64, dtype=mx.bfloat16, transposed_metadata=False):
    E, N, K, M = 4, 128, 256, 16
    x = mx.zeros((M, 1, K), dtype=dtype)
    w = mx.zeros((E, N, K * bits // 32), dtype=mx.uint32)
    shape = (E, K // group, N) if transposed_metadata else (E, N, K // group)
    s = mx.zeros(shape, dtype=dtype)
    idx = mx.zeros((M,), dtype=mx.uint32)
    rmap = mx.zeros((M,), dtype=mx.uint32)
    return x, w, s, s, idx, rmap


def test_supports_accepts_affine_q4_gs64_checkpoint_layout():
    x, w, s, b, idx, rmap = _operands()
    assert a8.supports(x, w, s, b, idx, 64, 4, "affine", rmap)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"bits": 8},  # Q8
        {"group": 32},  # GS32
        {"dtype": mx.float32},  # unsupported dtype
        {"transposed_metadata": True},  # [E, G, N], not the checkpoint layout
    ],
)
def test_supports_declines_unsupported_layouts(kwargs):
    x, w, s, b, idx, rmap = _operands(**kwargs)
    bits, group = kwargs.get("bits", 4), kwargs.get("group", 64)
    assert not a8.supports(x, w, s, b, idx, group, bits, "affine", rmap)
