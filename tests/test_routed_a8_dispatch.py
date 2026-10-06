# SPDX-License-Identifier: Apache-2.0
"""Routing of the routed-expert A8 Gate+Up: when it runs, and when it must not.

The kernel itself is covered by ``test_m5_gather_qmm_a8.py``. These tests pin
the dispatch contract, including that an eligible call really reaches the A8
kernel: a path that silently falls back to A16 produces plausible output, so
output checks alone cannot tell the two apart.
"""

import logging

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.models.switch_layers import SwitchGLU

from omlx.patches import m5_gather_qmm_a8 as a8
from omlx.patches import m5_gather_qmm_nax as nax
from omlx.patches import qwen35_moe_gate_up as gate_up
from omlx.patches import qwen35_moe_weighted_sum as weighted_sum_patch
from omlx.patches.moe_expert_offload import OffloadSwitchGLU

E, H, INTER, TOPK = 16, 256, 128, 10


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


@pytest.fixture(autouse=True)
def _restore_call_patch(monkeypatch):
    monkeypatch.delenv("OMLX_M5_GATHER_QMM_A8", raising=False)
    monkeypatch.setattr(a8, "_warned", set())
    orig = getattr(SwitchGLU, "_omlx_gate_up_original_call", SwitchGLU.__call__)
    yield
    SwitchGLU.__call__ = orig
    for attr in ("_omlx_gate_up_fused_call", "_omlx_gate_up_original_call"):
        if hasattr(SwitchGLU, attr):
            delattr(SwitchGLU, attr)
    gate_up._CALL_PATCHED = False


def _switch_glu(bits=4, group_size=64):
    sw = SwitchGLU(H, INTER, E)
    sw.set_dtype(mx.bfloat16)
    nn.quantize(
        sw,
        group_size=group_size,
        bits=bits,
        class_predicate=lambda p, m: hasattr(m, "to_quantized"),
    )
    mx.eval(sw.parameters())
    gate_up._fuse_one(sw)
    sw.eval()  # the sorted fast path is inference-only
    return sw


class _Layer(nn.Module):
    def __init__(self, switch_glu):
        super().__init__()
        self.switch_mlp = switch_glu


class _Backbone(nn.Module):
    def __init__(self, switch_glus):
        super().__init__()
        self.layers = [_Layer(sw) for sw in switch_glus]


# The family check reads the module path of the model class (as the gate+up
# fusion's own family check does).
_Backbone.__module__ = "mlx_vlm.models.qwen4_exp.qwen4_exp"


def _tagged(sw, min_tokens=128):
    """``sw`` as the engine opts it in (a one-layer model)."""
    assert a8.tag_routed_a8_modules(_Backbone([sw]), min_tokens) == 1
    return sw


def _inputs(batch, length, seed=0):
    x = mx.random.normal((batch, length, H), key=mx.random.key(seed)).astype(
        mx.bfloat16
    )
    ind = mx.random.randint(
        0, E, (batch, length, TOPK), key=mx.random.key(seed + 1)
    ).astype(mx.uint32)
    return x, ind


class _Spy:
    """Records every call of the A8 kernel entry point and what it returned."""

    def __init__(self, monkeypatch):
        self.results = []
        real = a8.sorted_gather_qmm_a8

        def spy(*args, **kwargs):
            out = real(*args, **kwargs)
            self.results.append(out)
            return out

        monkeypatch.setattr(a8, "sorted_gather_qmm_a8", spy)

    @property
    def ran(self) -> int:
        """Calls that produced an A8 result (not a decline)."""
        return sum(r is not None for r in self.results)


def _relative_error(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float(mx.abs(a - b).max() / mx.abs(b).max())


# -- the A8 path actually runs ------------------------------------------------


@needs_a8
def test_eligible_prefill_runs_the_a8_kernel_in_switch_glu_call(monkeypatch):
    """Below 1024 tokens the hook is ``SwitchGLU.__call__``."""
    gate_up._ensure_call_patch()
    sw = _tagged(_switch_glu())
    x, ind = _inputs(1, 200)
    spy = _Spy(monkeypatch)
    out = sw(x, ind)
    mx.eval(out)
    assert spy.ran == 1 and spy.results[0].shape[-1] == INTER

    # Not a silent A16 fallback: A16 gives different bits within the A8 error.
    setattr(sw, a8._TAG, None)
    ref = sw(x, ind)
    mx.eval(ref)
    assert spy.ran == 1
    assert not nax._bits_equal(out, ref)
    assert _relative_error(out, ref) < 0.05


def _weighted_sum(y, inv_order, scores):
    rows = y.reshape(y.shape[0], -1)[inv_order]
    return (rows.reshape(-1, TOPK, rows.shape[-1]).astype(mx.float32)
            * scores.reshape(-1, TOPK, 1)).sum(axis=1)


@needs_a8
def test_eligible_prefill_runs_the_a8_kernel_in_weighted_sum_path(monkeypatch):
    """From 1024 tokens the weighted-sum patch bypasses ``SwitchGLU.__call__``
    and has to reach the kernel by itself."""
    sw = _tagged(_switch_glu())
    x, ind = _inputs(1, 1200)
    scores = mx.softmax(mx.random.normal(ind.shape, key=mx.random.key(9)), axis=-1)
    spy = _Spy(monkeypatch)
    out = weighted_sum_patch._native_switch_weighted_sum(
        sw, x, ind, scores, _weighted_sum
    )
    mx.eval(out)
    assert spy.ran == 1

    setattr(sw, a8._TAG, None)
    ref = weighted_sum_patch._native_switch_weighted_sum(
        sw, x, ind, scores, _weighted_sum
    )
    mx.eval(ref)
    assert spy.ran == 1
    # the A8 result is what the path returns, not just something it computed
    assert not mx.array_equal(out, ref).item()
    assert _relative_error(out, ref) < 0.05


@needs_a8
def test_no_persistent_metadata_is_created(monkeypatch):
    gate_up._ensure_call_patch()
    sw = _tagged(_switch_glu())
    before = {k: id(v) for k, v in sw.gate_up_proj.items()}
    x, ind = _inputs(1, 200)
    mx.eval(sw(x, ind))
    assert {k: id(v) for k, v in sw.gate_up_proj.items()} == before
    extra = [
        name
        for proj in (sw.gate_up_proj, sw.down_proj)
        for name in vars(proj)
        if name.startswith("_omlx") and isinstance(getattr(proj, name), mx.array)
    ]
    assert extra == []


# -- everything else keeps A16 -----------------------------------------------


@needs_a8
def test_module_that_was_not_opted_in_keeps_a16(monkeypatch):
    gate_up._ensure_call_patch()
    sw = _switch_glu()  # a model loaded with the A8 setting off is never tagged
    x, ind = _inputs(1, 200)
    spy = _Spy(monkeypatch)
    mx.eval(sw(x, ind))
    assert spy.results == []


@needs_a8
@pytest.mark.parametrize("length", [1, 16, 64, 127])
def test_short_sequences_keep_a16(length, monkeypatch):
    """Below 128 tokens: single-token decode, verify windows, short prefill."""
    gate_up._ensure_call_patch()
    sw = _tagged(_switch_glu())
    x, ind = _inputs(1, length)
    spy = _Spy(monkeypatch)
    mx.eval(sw(x, ind))
    assert spy.results == []


@needs_a8
def test_batched_decode_keeps_a16(monkeypatch):
    """128 sequences of one token are 128 tokens but a sequence length of 1:
    the sorted hook is reached and must decline."""
    gate_up._ensure_call_patch()
    sw = _tagged(_switch_glu())
    reached = []
    real = gate_up.try_routed_a8

    def hook(*args, **kwargs):
        out = real(*args, **kwargs)
        reached.append(out is not None)
        return out

    monkeypatch.setattr(gate_up, "try_routed_a8", hook)
    spy = _Spy(monkeypatch)
    x, ind = _inputs(128, 1)
    mx.eval(sw(x, ind))
    assert reached == [False]
    assert spy.results == []


@needs_a8
def test_model_min_tokens_can_raise_the_floor_but_not_lower_it(monkeypatch):
    gate_up._ensure_call_patch()
    spy = _Spy(monkeypatch)
    x, ind = _inputs(1, 200)

    low = _tagged(_switch_glu(), min_tokens=1)  # floor stays 128
    mx.eval(low(x, ind))
    assert spy.ran == 1
    x_short, ind_short = _inputs(1, 100)
    mx.eval(low(x_short, ind_short))
    assert spy.ran == 1

    high = _tagged(_switch_glu(), min_tokens=512)
    mx.eval(high(x, ind))
    assert spy.ran == 1


@needs_a8
@pytest.mark.parametrize("bits,group_size", [(8, 64), (4, 32)])
def test_unsupported_quantization_keeps_a16(bits, group_size, monkeypatch):
    gate_up._ensure_call_patch()
    sw = _tagged(_switch_glu(bits=bits, group_size=group_size))
    x, ind = _inputs(1, 200)
    spy = _Spy(monkeypatch)
    mx.eval(sw(x, ind))
    assert spy.results == []


@needs_a8
def test_clamped_activation_keeps_a16(monkeypatch):
    gate_up._ensure_call_patch()
    sw = _tagged(_switch_glu())
    clamped = type("Glm5NextClampedSwiGLU", (), {"limit": 7.0})
    clamped.__module__ = "mlx_vlm.models.glm5_next.language"
    sw.activation = clamped()
    x, ind = _inputs(1, 200)
    spy = _Spy(monkeypatch)
    sw(x, ind)
    assert spy.results == []


@needs_a8
def test_expert_offload_module_keeps_a16(monkeypatch):
    """An ``OffloadSwitchGLU`` that somehow carried the opt-in and a fused
    projection is still declined: its weights are a remapped resident table."""
    sw = _tagged(_switch_glu())

    class _Offload(OffloadSwitchGLU):
        def __init__(self, source):
            nn.Module.__init__(self)
            self.gate_up_proj = source.gate_up_proj
            self.down_proj = source.down_proj
            self.activation = source.activation
            setattr(self, a8._TAG, 128)

    offload = _Offload(sw)
    spy = _Spy(monkeypatch)
    x_tok = mx.random.normal((200, 1, H)).astype(mx.bfloat16)
    row_map = mx.arange(200 * TOPK, dtype=mx.uint32) % 200
    idx = mx.sort(mx.random.randint(0, E, (200 * TOPK,)).astype(mx.uint32))
    assert a8.try_routed_a8(offload, (x_tok, row_map), idx) is None
    assert spy.results == []


@needs_a8
def test_disabled_module_tags_nothing(monkeypatch):
    monkeypatch.setenv("OMLX_M5_GATHER_QMM_A8", "0")
    sw = _switch_glu()
    assert a8.tag_routed_a8_modules(_Backbone([sw]), 128) == 0
    assert getattr(sw, a8._TAG, None) is None


# -- MTP exclusion ------------------------------------------------------------


class _Draft(nn.Module):
    def __init__(self, switch_glu):
        super().__init__()
        self.layers = [_Layer(switch_glu)]


class _LanguageWithMtpAttr(_Backbone):
    def __init__(self, backbone, draft):
        super().__init__(backbone)
        self.mtp = _Draft(draft)


class _LanguageWithGetter(_Backbone):
    """The draft head lives under a name that does not start with ``mtp``."""

    def __init__(self, backbone, draft):
        super().__init__(backbone)
        self.draft_head = _Draft(draft)

    def get_mtp_module(self):
        return self.draft_head


class _Wrapper(nn.Module):
    def __init__(self, language):
        super().__init__()
        self.language_model = language


for _cls in (_LanguageWithMtpAttr, _LanguageWithGetter, _Wrapper):
    _cls.__module__ = _Backbone.__module__


@pytest.mark.parametrize("variant", ["mtp_attr", "getter_other_name", "wrapped"])
def test_mtp_draft_head_is_excluded_by_identity(variant):
    backbone, draft = [_switch_glu(), _switch_glu()], _switch_glu()
    if variant == "getter_other_name":
        model = _LanguageWithGetter(backbone, draft)
    else:
        model = _LanguageWithMtpAttr(backbone, draft)
        if variant == "wrapped":
            model = _Wrapper(model)
    assert a8.tag_routed_a8_modules(model, 128) == 2
    assert all(getattr(sw, a8._TAG, None) == 128 for sw in backbone)
    assert getattr(draft, a8._TAG, None) is None


def test_module_attached_after_tagging_stays_a16():
    sw = _switch_glu()
    model = _Backbone([sw])
    a8.tag_routed_a8_modules(model, 128)
    late = _switch_glu()
    model.layers.append(_Layer(late))
    assert getattr(sw, a8._TAG, None) == 128
    assert getattr(late, a8._TAG, None) is None


def test_other_model_families_are_not_tagged():
    """Only families whose routed Gate+Up was measured are opted in, although
    other Qwen MoE models can enable ``qwen35_oq_a8_enabled``."""

    class _OtherFamily(_Backbone):
        pass

    _OtherFamily.__module__ = "mlx_lm.models.qwen3_5_moe"
    sw = _switch_glu()
    assert a8.tag_routed_a8_modules(_OtherFamily([sw]), 128) == 0
    assert getattr(sw, a8._TAG, None) is None


def test_unfused_switch_glu_is_not_tagged():
    unfused = SwitchGLU(H, INTER, E)
    assert a8.tag_routed_a8_modules(_Backbone([unfused]), 128) == 0


def test_the_per_model_a8_setting_tags_the_routed_modules(monkeypatch):
    """``apply_qwen35_oq_a8_patch`` (called for ``qwen35_oq_a8_enabled``) is the
    only place that opts a model in, with the model's own ``min_tokens``."""
    from omlx.patches import qwen35_oq_a8

    monkeypatch.setattr(qwen35_oq_a8, "_MLP_PATCHED", True)
    monkeypatch.setattr(qwen35_oq_a8, "_GDN_REGISTERED", True)
    tagged = _switch_glu()
    other_model_module = _switch_glu()
    qwen35_oq_a8.apply_qwen35_oq_a8_patch(_Backbone([tagged]), min_tokens=256)
    assert getattr(tagged, a8._TAG, None) == 256
    assert getattr(other_model_module, a8._TAG, None) is None


# -- failures are visible, once ----------------------------------------------


@needs_a8
def test_exception_falls_back_and_warns_once_per_type(monkeypatch, caplog):
    sw = _tagged(_switch_glu())
    x_tok = mx.random.normal((200, 1, H)).astype(mx.bfloat16)
    row_map = mx.arange(200 * TOPK, dtype=mx.uint32) % 200
    idx = mx.sort(mx.random.randint(0, E, (200 * TOPK,)).astype(mx.uint32))
    error = [RuntimeError("boom")]

    def broken(*args, **kwargs):
        raise error[0]

    monkeypatch.setattr(a8, "sorted_gather_qmm_a8", broken)
    with caplog.at_level(logging.WARNING, logger=a8.logger.name):
        for _ in range(3):
            assert a8.try_routed_a8(sw, (x_tok, row_map), idx) is None
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 1
        assert "RuntimeError" in warnings[0].getMessage()
        error[0] = ValueError("other")
        for _ in range(2):
            assert a8.try_routed_a8(sw, (x_tok, row_map), idx) is None
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert len(warnings) == 2
