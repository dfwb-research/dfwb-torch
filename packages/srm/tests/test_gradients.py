"""Module construction and gradient tests for `SRMConv2d`: gradcheck
(float64), buffer/parameter registration, the trainable zero-DC invariant,
`out_channels` per mode, a state-dict round trip, and that `forward` is
stateless (no cache, no mutation) so it plays well with `inference_mode`,
`torch.jit.trace` and plain buffer edits/loads. Device/dtype and
torch.compile parity have their own files: `test_device_dtype.py` and
`test_compile.py`.
"""

import copy

import pytest
import torch
from dfwb_torch_srm.modules import TLU, SRMConv2d


def test_gradcheck_wrt_input() -> None:
    """`SRMConv2d` is linear (conv) + clamp (TLU); gradcheck must pass w.r.t.
    the input in float64.

    Truncation is disabled here (`truncate=None`): TLU's clamp has a kink at
    its threshold, and with the default 0-255-unit threshold that kink sits
    at an absolute value of only ``3/255 ~= 0.0118``, so almost every entry
    of a standard-normal input would saturate and sit close enough to the
    boundary that finite differences could straddle it. Disabling truncation
    isolates the part of the module gradcheck is meant to certify here: the
    convolution's linearity w.r.t. the input. TLU's own clamp gradient is a
    single well-understood PyTorch primitive, not new code.
    """
    torch.manual_seed(0)
    layer = SRMConv2d(in_channels=2, bank="square3", mode="sum", truncate=None)
    x = torch.randn(1, 2, 5, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(layer, (x,))


def test_fixed_kernels_are_buffers_not_parameters() -> None:
    """`trainable=False` (the default): `weight` is a buffer, contributes no
    gradient, and does not show up in `.parameters()`."""
    from torch import nn

    layer = SRMConv2d(trainable=False)
    assert "weight" in dict(layer.named_buffers())
    assert "weight" not in dict(layer.named_parameters())
    assert list(layer.parameters()) == []
    assert not isinstance(layer.weight, nn.Parameter)
    assert not layer.weight.requires_grad


def test_trainable_kernels_stay_zero_dc_after_steps() -> None:
    """After optimiser steps, every kernel must still sum to (numerically)
    zero, so training can never turn the layer into a low-pass filter.

    5 SGD steps at an aggressive lr=1.0 push the *raw*, pre-parametrisation
    parameter hard in a random direction (a random, non-degenerate loss).
    The test checks both that the raw parameter actually moved (so this
    isn't vacuously true because nothing trained) and that the parametrised
    `weight` the module actually uses is still zero-sum after every step.
    """
    torch.manual_seed(0)
    layer = SRMConv2d(in_channels=3, trainable=True)  # defaults: srm30, depthwise
    raw_before = layer.parametrizations.weight.original.detach().clone()

    optimizer = torch.optim.SGD(layer.parameters(), lr=1.0)
    x = torch.randn(2, 3, 9, 9)
    for _ in range(5):
        optimizer.zero_grad()
        y = layer(x)
        loss = (y * torch.randn(y.shape)).sum()  # a random, non-degenerate loss
        loss.backward()
        optimizer.step()

    raw_after = layer.parametrizations.weight.original.detach()
    assert not torch.allclose(raw_before, raw_after)  # training actually moved the raw parameter

    kernel_sums = layer.weight.detach().sum(dim=(-2, -1))
    assert torch.allclose(kernel_sums, torch.zeros_like(kernel_sums), atol=1e-5)


def test_trainable_kernels_stay_zero_dc_at_init() -> None:
    """Before any training, the parametrised weight is already zero-sum."""
    layer = SRMConv2d(bank="rgbn3", trainable=True)
    kernel_sums = layer.weight.detach().sum(dim=(-2, -1))
    assert torch.allclose(kernel_sums, torch.zeros_like(kernel_sums), atol=1e-6)


def test_per_channel_needs_k_equal_c_at_construction() -> None:
    """Unlike the functional API (which only checks at call time, since it
    doesn't know `in_channels` ahead of time), the module knows both `K` and
    `in_channels` at construction, so it validates `mode="per-channel"`
    eagerly and raises a clear `ValueError`."""
    with pytest.raises(ValueError, match=r"K == C.*K=30.*C=3"):
        SRMConv2d(in_channels=3, bank="srm30", mode="per-channel")


def test_gray_needs_three_channels_at_construction() -> None:
    """Unlike `mode="per-channel"`'s `K == C` check (which the functional API
    also can't do ahead of time), `mode="gray"` needs exactly 3 input
    channels; the module already knows `in_channels` at construction, so it
    validates this eagerly rather than waiting for the first forward pass
    (the functional API, which only sees the actual tensor's channel count,
    still validates this at call time)."""
    with pytest.raises(ValueError, match=r"gray.*C=4"):
        SRMConv2d(in_channels=4, bank="srm30", mode="gray")


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"input_scale": "0-100"}, r"input_scale.*'0-1'.*'0-255'"),
        ({"padding": "full"}, r"padding.*'same'.*'valid'"),
        ({"padding_mode": "circular"}, r"padding_mode.*'reflect'.*'zeros'.*'replicate'"),
        ({"truncate": -1.0}, r"truncate.*-1\.0"),
    ],
)
def test_srmconv2d_invalid_options_raise_at_construction(
    kwargs: dict[str, object], match: str
) -> None:
    """`SRMConv2d` validates its string-choice and numeric options eagerly,
    the same way `srm_conv2d` does, rather than accepting an invalid option
    silently and misbehaving (or raising a confusing error) on the first
    forward pass."""
    with pytest.raises(ValueError, match=match):
        SRMConv2d(in_channels=3, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"input_scale": "0-100"}, r"input_scale.*'0-1'.*'0-255'"),
        ({"threshold": -1.0}, r"threshold.*-1\.0"),
        ({"threshold": 0.0}, r"threshold.*0\.0"),
    ],
)
def test_tlu_invalid_options_raise_at_construction(kwargs: dict[str, object], match: str) -> None:
    base = {"threshold": 3.0, "input_scale": "0-1"}
    base.update(kwargs)
    with pytest.raises(ValueError, match=match):
        TLU(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mode", "bank", "in_channels", "expected"),
    [
        ("depthwise", "srm30", 3, 90),
        ("sum", "srm30", 3, 30),
        ("gray", "srm30", 3, 30),
        ("per-channel", "rgbn3", 3, 3),
        ("depthwise", "square3", 1, 1),
    ],
)
def test_out_channels_per_mode(mode: str, bank: str, in_channels: int, expected: int) -> None:
    layer = SRMConv2d(in_channels=in_channels, bank=bank, mode=mode)
    assert layer.out_channels == expected
    x = torch.zeros(1, in_channels, 9, 9)
    assert layer(x).shape[1] == expected


@pytest.mark.parametrize("trainable", [False, True])
def test_state_dict_round_trip(trainable: bool) -> None:
    torch.manual_seed(0)
    src = SRMConv2d(in_channels=3, bank="rgbn3", mode="sum", trainable=trainable)
    dst = SRMConv2d(in_channels=3, bank="rgbn3", mode="sum", trainable=trainable)
    if trainable:
        with torch.no_grad():
            src.parametrizations.weight.original.add_(0.5)  # perturb before the round trip

    dst.load_state_dict(src.state_dict())

    assert torch.equal(dst.weight, src.weight)
    x = torch.randn(1, 3, 8, 8)
    assert torch.equal(dst(x), src(x))


@pytest.mark.parametrize("mode", ["depthwise", "sum", "gray", "per-channel"])
def test_forward_rejects_input_with_wrong_channel_count(mode: str) -> None:
    """Every mode must reject an input whose channel count does not match
    the declared `in_channels`. `gray` and `per-channel` already raised (via
    the luminance and `K == C` checks respectively), but `depthwise` and
    `sum` would otherwise silently compute an output whose channel count no
    longer matches `out_channels`, since their weight only depends on the
    *actual* input's channel count, not on the module's declared
    `in_channels`."""
    bank = "rgbn3" if mode == "per-channel" else "srm30"
    layer = SRMConv2d(in_channels=3, bank=bank, mode=mode)
    x = torch.zeros(1, 5, 8, 8)
    with pytest.raises(ValueError, match=r"in_channels=3.*C=5"):
        layer(x)


@pytest.mark.parametrize(("mode", "bank"), [("gray", "srm30"), ("per-channel", "rgbn3")])
def test_inference_mode_then_backward_wrt_input(mode: str, bank: str) -> None:
    """`forward` must read no cached state from an earlier call: a call
    inside `torch.inference_mode()` must not leave anything behind that
    later poisons an ordinary, grad-tracked call. Regression: a per-call
    weight cache built its tensor under whatever mode was active on the
    *first* call and then reused it on every later call regardless of mode,
    so a `gray`/`per-channel` layer warmed up under `inference_mode()`
    raised `RuntimeError: Inference tensors cannot be saved for backward`
    the first time it was then used in a normal backward pass."""
    layer = SRMConv2d(in_channels=3, bank=bank, mode=mode, truncate=None)
    with torch.inference_mode():
        layer(torch.randn(1, 3, 8, 8))
    x = torch.randn(1, 3, 8, 8, requires_grad=True)
    y = layer(x)
    y.sum().backward()
    assert x.grad is not None


def test_jit_trace_fresh_module() -> None:
    """`torch.jit.trace` runs `forward` twice and checks the two graphs
    match. Regression: a per-call weight cache made the first call build and
    store a tensor and the second call reuse it, recording two different
    graphs (`Graphs differed across invocations!`) for a module that had
    never been called before tracing."""
    layer = SRMConv2d(3, mode="gray")
    x = torch.randn(2, 3, 16, 16)
    traced = torch.jit.trace(layer, x)
    assert torch.allclose(traced(x), layer(x))


def test_forward_does_not_mutate_module_state() -> None:
    """`forward` is a pure function of `self` and `x`: it must add no new
    attribute and change no existing one. Regression: a per-call weight
    cache was a plain dict attribute that `forward` mutated in place on
    every new (dtype, device) it saw."""

    def snapshot(value: object) -> object:
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, dict):
            return {k: snapshot(v) for k, v in value.items()}
        return copy.deepcopy(value)

    def snapshots_equal(a: object, b: object) -> bool:
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict):
            return (
                isinstance(b, dict)
                and a.keys() == b.keys()
                and all(snapshots_equal(a[k], b[k]) for k in a)
            )
        return bool(a == b)

    layer = SRMConv2d(in_channels=3, bank="srm30", mode="sum")
    before = {k: snapshot(v) for k, v in vars(layer).items()}
    layer(torch.randn(1, 3, 8, 8))
    after = vars(layer)

    assert before.keys() == after.keys(), "forward must add or remove no attribute"
    for key, before_value in before.items():
        assert snapshots_equal(before_value, after[key]), f"forward changed {key!r}"


def test_editing_fixed_weight_in_place_changes_the_output() -> None:
    """The `weight` buffer is the sole source of truth for a fixed layer:
    `forward` must read it fresh every call, not some other, separately
    resolved value."""
    layer = SRMConv2d(in_channels=1, bank="square3", mode="sum", truncate=None)
    x = torch.randn(1, 1, 8, 8)
    before = layer(x)
    with torch.no_grad():
        layer.weight.mul_(2)
    after = layer(x)
    assert not torch.equal(before, after)


def test_load_state_dict_weight_changes_the_output() -> None:
    """Loading a different `weight` via `load_state_dict` must change what
    `forward` computes, the same as editing the buffer directly."""
    layer = SRMConv2d(in_channels=1, bank="square3", mode="sum", truncate=None)
    x = torch.randn(1, 1, 8, 8)
    state = layer.state_dict()
    state["weight"] = state["weight"] * 0
    layer.load_state_dict(state)
    assert torch.equal(layer(x), torch.zeros_like(layer(x)))


def test_rc1_style_float32_checkpoint_loads() -> None:
    """A fixed layer's `weight` buffer is float64 (see the class docstring),
    but a checkpoint saved by the original 0.1.0rc1 (whose buffer was
    float32) must still load: `load_state_dict` copies the source values
    into the destination buffer, casting to the destination's own dtype."""
    from dfwb_torch_srm.kernels import srm_kernels

    layer = SRMConv2d(in_channels=3, bank="srm30", mode="sum")
    assert layer.weight.dtype == torch.float64
    rc1_weight = srm_kernels("srm30", dtype=torch.float32)  # rc1's construction dtype
    layer.load_state_dict({"weight": rc1_weight})
    assert layer.weight.dtype == torch.float64  # the buffer keeps its own dtype
    layer(torch.randn(1, 3, 8, 8))  # does not raise


@pytest.mark.parametrize("in_channels", [0, -1])
def test_in_channels_must_be_at_least_one(in_channels: int) -> None:
    with pytest.raises(ValueError, match=r"in_channels"):
        SRMConv2d(in_channels=in_channels)


def test_args_after_in_channels_are_keyword_only() -> None:
    """Matches `srm_conv2d`'s own keyword-only style for every option after
    the leading positional argument."""
    with pytest.raises(TypeError):
        SRMConv2d(3, "srm30")  # type: ignore[misc]


def test_unknown_bank_raises_keyerror_at_construction() -> None:
    with pytest.raises(KeyError):
        SRMConv2d(bank="nope")


def test_non_string_kernel_name_raises_typeerror_at_construction() -> None:
    with pytest.raises(TypeError):
        SRMConv2d(bank=[1, 2])  # type: ignore[list-item]


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int64])
@pytest.mark.parametrize("trainable", [False, True])
def test_forward_rejects_non_floating_input(dtype: torch.dtype, trainable: bool) -> None:
    """Regression: a non-floating input previously either raised a
    confusing internal error (fixed kernels, naming `srm_kernels`, not the
    call the user actually made) or silently computed garbage in integer
    arithmetic (trainable kernels, no error at all)."""
    layer = SRMConv2d(in_channels=3, bank="srm30", mode="sum", trainable=trainable)
    x = (torch.rand(1, 3, 8, 8) * 255).to(dtype)
    with pytest.raises(TypeError, match=r"floating-point tensor.*x\.float\(\)"):
        layer(x)


def test_repr_includes_config() -> None:
    layer = SRMConv2d(in_channels=3, bank="rgbn3", mode="sum", trainable=True)
    text = repr(layer)
    assert "in_channels=3" in text
    assert "out_channels=3" in text
    assert "mode='sum'" in text
    assert "trainable=True" in text

    tlu_text = repr(TLU(threshold=2.0, input_scale="0-255"))
    assert "threshold=2.0" in tlu_text
    assert "input_scale='0-255'" in tlu_text
