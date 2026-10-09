# SPDX-License-Identifier: Apache-2.0
"""Q8 W8A8 prefill over the checkpoint's own scale/bias layout (no [K/64, N] copy).

The native-metadata kernel must be bit-identical to the group-major kernel it replaces for Q8
GS64 and must not leave any second copy of the group metadata.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

GROUP_SIZE = 64


def _kernels():
    try:
        from omlx.custom_kernels.qwen35_prefill import fast
    except Exception:
        return None
    if not fast.oq_a8_available():
        return None
    return fast


requires_kernels = pytest.mark.skipif(
    _kernels() is None,
    reason="oQ A8 NAX kernels are unavailable (no native build or no tensor units)",
)


def _raw_q8(n, k, dtype, seed):
    """Random Q8 tensors, with the boundary codes 0, 127, 128 and 255 forced in."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 256, size=(n, k), dtype=np.uint8)
    codes[:, :4] = [0, 127, 128, 255]
    codes[0, :] = 255
    codes[1, :] = 0
    codes[2, :] = 128
    codes[3, :] = 127
    packed = mx.array(codes.view("<u4").copy())
    scales = mx.array(
        rng.uniform(0.001, 0.05, (n, k // GROUP_SIZE)).astype(np.float32), dtype
    )
    biases = mx.array(
        rng.uniform(-3.0, 1.0, (n, k // GROUP_SIZE)).astype(np.float32), dtype
    )
    return codes, packed, scales, biases


def _bits(a):
    return np.array(a.astype(mx.float32)).view(np.uint32)


def _quantized_linear(in_dim, out_dim, bits):
    import mlx.nn as nn

    linear = nn.QuantizedLinear(
        in_dim, out_dim, bias=False, group_size=GROUP_SIZE, bits=bits
    )
    linear.set_dtype(mx.float16)
    mx.eval(linear.parameters())
    return linear


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------


@requires_kernels
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("act_mode", [0, 1])
@pytest.mark.parametrize("m", [1, 3, 31, 33, 127, 128, 129, 1000, 1024, 1025, 1500])
def test_native_metadata_is_bit_identical_to_transposed(dtype, act_mode, m):
    """Same Stage A, same arithmetic: every output element must match bit for bit.

    m covers a single row, M % 4 != 0 (scalar token-metadata path), partial tiles, and
    both sides of the 1024-row boundary where the host switches tile.
    """
    fast = _kernels()
    n, k = 192, 512
    _, packed, scales, biases = _raw_q8(n, k, dtype, seed=m + 7 * act_mode)
    rng = np.random.default_rng(m)
    x = mx.array((rng.standard_normal((m, k)) * 0.5).astype(np.float32), dtype)
    qa, sa, ra = fast.qwen35_oq_a8_stage_a_natural(x, act_mode)

    want = fast.qwen35_oq_a8_qmm_t(
        qa,
        sa,
        ra,
        packed,
        mx.contiguous(scales.T),
        mx.contiguous(biases.T),
        8,
        act_mode,
        806,
    )
    got = fast.qwen35_oq_a8_qmm_t(
        qa, sa, ra, packed, scales, biases, 8, act_mode, 806, native_meta=True
    )
    mx.eval(want, got)
    assert got.shape == want.shape and got.dtype == want.dtype
    np.testing.assert_array_equal(_bits(got), _bits(want))


@requires_kernels
@pytest.mark.parametrize("act_mode", [0, 1])
def test_native_metadata_matches_independent_affine_reference(act_mode):
    """Anchor to the checkpoint format, not only to the other kernel."""
    fast = _kernels()
    m, n, k = 130, 128, 256
    codes, packed, scales, biases = _raw_q8(n, k, mx.float16, seed=11)
    rng = np.random.default_rng(12)
    x = mx.array((rng.standard_normal((m, k)) * 0.5).astype(np.float32), mx.float16)
    qa, sa, ra = fast.qwen35_oq_a8_stage_a_natural(x, act_mode)
    got = fast.qwen35_oq_a8_qmm_t(
        qa, sa, ra, packed, scales, biases, 8, act_mode, 806, native_meta=True
    )
    mx.eval(got, qa, sa)

    s = np.array(scales.astype(mx.float32)).astype(np.float64).repeat(GROUP_SIZE, 1)
    b = np.array(biases.astype(mx.float32)).astype(np.float64).repeat(GROUP_SIZE, 1)
    deq = s * codes.astype(np.float64) + b
    qa64 = np.array(qa).astype(np.float64)
    sa64 = np.array(sa).astype(np.float64)
    want = np.zeros((m, n))
    for g in range(k // GROUP_SIZE):
        part = qa64[:, g * 64 : (g + 1) * 64] @ deq[:, g * 64 : (g + 1) * 64].T
        want += sa64[g][:, None] * part if act_mode == 1 else part
    if act_mode == 0:
        want *= sa64[:, None]
    np.testing.assert_allclose(
        np.array(got.astype(mx.float32)), want, rtol=2**-7, atol=2e-3
    )


@requires_kernels
def test_native_metadata_rejects_wrong_layout_and_unsupported_modes():
    fast = _kernels()
    n, k, m = 128, 256, 64
    _, packed, scales, biases = _raw_q8(n, k, mx.float16, seed=1)
    x = mx.zeros((m, k), mx.float16)
    qa, sa, ra = fast.qwen35_oq_a8_stage_a_natural(x, 0)
    # A group-major copy handed to the native kernel is a shape error, not silent garbage.
    with pytest.raises(Exception, match="incompatible"):
        fast.qwen35_oq_a8_qmm_t(
            qa,
            sa,
            ra,
            packed,
            mx.contiguous(scales.T),
            mx.contiguous(biases.T),
            8,
            0,
            806,
            native_meta=True,
        )
    # Only the 806 tile family exists for the native kernel.
    with pytest.raises(Exception, match="806"):
        fast.qwen35_oq_a8_qmm_t(
            qa, sa, ra, packed, scales, biases, 8, 0, 800, native_meta=True
        )
    # Q4 has its own layout and is out of scope.
    with pytest.raises(Exception, match="Q8"):
        fast.qwen35_oq_a8_qmm_t(
            qa, sa, ra, packed, scales, biases, 4, 0, 806, native_meta=True
        )


# --------------------------------------------------------------------------
# Routing and memory
# --------------------------------------------------------------------------


@pytest.fixture
def planner_available(monkeypatch):
    """Let the planner classify projections on hosts without the native extension (CI)."""
    from omlx.custom_kernels.qwen35_prefill import fast

    monkeypatch.setattr(fast, "oq_a8_available", lambda: True)


@pytest.mark.parametrize("bits, expected", [(8, True), (4, False), (5, False)])
def test_only_q8_reads_the_checkpoint_layout(planner_available, bits, expected):
    from omlx.patches import qwen35_oq_a8 as dispatch

    linear = _quantized_linear(256, 128, bits)
    plan = dispatch.classify_linear(linear)
    assert plan is not None and plan.native_meta is expected
    assert plan.variant == (806 if bits in (4, 8) else 800)


def _stage(plan, m=130, k=256):
    from omlx.patches import qwen35_oq_a8 as dispatch

    return dispatch.StageA(
        qa=mx.zeros((m, k), mx.int8),
        sa=mx.zeros((k // GROUP_SIZE, m), mx.float32),
        ra=mx.zeros((k // GROUP_SIZE, m), mx.int16),
        act_mode=plan.act_mode,
        layout=plan.layout,
    )


def _recording_kernel(monkeypatch):
    from omlx.custom_kernels.qwen35_prefill import fast

    calls = []

    def recorder(qa, sa, ra, w, s, b, bits, act_mode, variant, **kw):
        calls.append((w, s, b, bits, variant, kw))
        return mx.zeros((qa.shape[0], w.shape[0]), mx.float16)

    monkeypatch.setattr(fast, "qwen35_oq_a8_qmm_t", recorder)
    return calls


def test_q8_apply_plan_hands_the_checkpoint_arrays_to_the_kernel(
    planner_available, monkeypatch
):
    """The layer's own [N, K/64] arrays and the native flag reach the kernel; no copy is cached."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    calls = _recording_kernel(monkeypatch)
    linear = _quantized_linear(256, 128, 8)
    plan = dispatch.classify_linear(linear)
    dispatch.apply_plan(linear, _stage(plan), plan)
    w, s, b, bits, variant, kw = calls[0]
    assert w is linear.weight and s is linear.scales and b is linear.biases
    assert s.shape == (128, 256 // GROUP_SIZE)
    assert (bits, variant) == (8, 806)
    assert kw.get("native_meta") is True and not kw.get("packed")
    assert getattr(linear, dispatch._PREPARED_ATTR, None) is None


@pytest.mark.parametrize("bits", [4, 5])
def test_q4_q5_apply_plan_keeps_the_group_major_copy(
    planner_available, monkeypatch, bits
):
    from omlx.patches import qwen35_oq_a8 as dispatch

    calls = _recording_kernel(monkeypatch)
    linear = _quantized_linear(256, 128, bits)
    plan = dispatch.classify_linear(linear)
    dispatch.apply_plan(linear, _stage(plan), plan)
    w, s, b, got_bits, variant, kw = calls[0]
    assert s.shape == (256 // GROUP_SIZE, 128) and b.shape == s.shape
    assert got_bits == bits and not kw.get("native_meta")
    assert getattr(linear, dispatch._PREPARED_ATTR, None) is not None


@requires_kernels
def test_q8_projection_allocates_no_metadata_bytes():
    """Active memory across a routed projection grows by 0 bytes, not by 2*N*G*itemsize.

    The explicit group-major copy is the negative control: it proves the measurement can see one.
    """
    from omlx.patches import qwen35_oq_a8 as dispatch

    n, k, m = 2048, 4096, 256
    expected_copy = 2 * n * (k // GROUP_SIZE) * 2  # scales + biases, 16-bit
    linear = _quantized_linear(k, n, 8)
    x = mx.array(
        np.random.default_rng(0).standard_normal((m, k)).astype(np.float32), mx.float16
    )
    mx.eval(x)
    mx.synchronize()
    before = mx.get_active_memory()
    plan = dispatch.classify_linear(linear)
    assert plan is not None and plan.native_meta
    y = dispatch.apply_plan(
        linear, dispatch.stage_a(x, plan.act_mode, plan.layout), plan
    )
    mx.eval(y)
    mx.synchronize()
    del y
    mx.synchronize()
    routed = mx.get_active_memory() - before
    assert routed < expected_copy // 8

    before = mx.get_active_memory()
    copy = dispatch._prepared_weights(linear)  # the group-major copy, built on purpose
    mx.eval(*copy)
    mx.synchronize()
    assert mx.get_active_memory() - before >= expected_copy


@requires_kernels
@pytest.mark.parametrize("m", [130, 1024, 1500])
def test_q8_projection_matches_the_group_major_kernel(m):
    """Through the dispatch layer, the routed output equals the group-major kernel's bit for bit."""
    from omlx.patches import qwen35_oq_a8 as dispatch

    fast = _kernels()
    n, k = 256, 512
    rng = np.random.default_rng(m)
    x = mx.array((rng.standard_normal((m, k)) * 0.5).astype(np.float32), mx.float16)
    linear = _quantized_linear(k, n, 8)
    packed, scales, biases = mx.quantize(
        mx.array(rng.standard_normal((n, k)).astype(np.float32) * 0.05, mx.float16),
        group_size=GROUP_SIZE,
        bits=8,
        mode="affine",
    )
    linear.weight = packed
    linear.scales = scales.astype(mx.float16)
    linear.biases = biases.astype(mx.float16)
    plan = dispatch.classify_linear(linear)
    assert plan is not None and plan.native_meta
    stage = dispatch.stage_a(x, plan.act_mode, plan.layout)
    got = dispatch.apply_plan(linear, stage, plan)
    want = fast.qwen35_oq_a8_qmm_t(
        stage.qa,
        stage.sa,
        stage.ra,
        linear.weight,
        mx.contiguous(linear.scales.T),
        mx.contiguous(linear.biases.T),
        8,
        plan.act_mode,
        plan.variant,
    )
    mx.eval(got, want)
    np.testing.assert_array_equal(_bits(got), _bits(want))
