"""Shape and behaviour tests for srm's functional API (`srm_conv2d`, `tlu`).

Covers every mode x shape x padding combination (batch 1, an odd/non-square
image, a 1x1 image with `padding="same"`, and an empty batch), the
input-scale invariance for truncation, TLU clamping, the mode-specific
`ValueError`s, argument validation (`input_scale`, `padding`,
`padding_mode`, `truncate`, and `padding="valid"` on a too-small image),
and that `srm_conv2d` never mutates its input.
"""

import inspect

import pytest
import torch
from dfwb_torch_srm.functional import srm_conv2d, tlu

_SHAPES = [
    (1, 3, 7, 5),  # odd, non-square
    (2, 3, 1, 1),  # 1x1: reflect padding is impossible for "same"
    (0, 3, 8, 8),  # empty batch
    (4, 3, 16, 16),  # a typical size
]
_MODES = ("depthwise", "sum", "gray", "per-channel")
_PADDINGS = ("same", "valid")

# "per-channel" needs K == C; rgbn3 has exactly 3 kernels, matching these
# C=3 shapes. The other modes use the default 30-kernel bank.
_BANK_FOR_MODE = {"depthwise": "srm30", "sum": "srm30", "gray": "srm30", "per-channel": "rgbn3"}
_K_FOR_MODE = {"depthwise": 30, "sum": 30, "gray": 30, "per-channel": 3}


def _expected_out_channels(mode: str, c: int, k: int) -> int:
    if mode == "depthwise":
        return c * k
    if mode in ("sum", "gray"):
        return k
    return c  # per-channel: K == C, so this equals k too


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("padding", _PADDINGS)
def test_shapes_matrix(mode: str, padding: str, shape: tuple[int, int, int, int]) -> None:
    b, c, h, w = shape
    bank = _BANK_FOR_MODE[mode]
    k = _K_FOR_MODE[mode]
    x = torch.zeros(b, c, h, w)

    if padding == "valid" and (h < 5 or w < 5):
        # A 5x5 kernel does not fit an image smaller than 5x5 with no
        # padding; srm_conv2d raises its own clear ValueError rather than
        # letting F.conv2d fail with an opaque RuntimeError.
        with pytest.raises(ValueError, match=r"padding='valid'.*H=\d+.*W=\d+"):
            srm_conv2d(x, bank=bank, mode=mode, padding=padding)
        return

    y = srm_conv2d(x, bank=bank, mode=mode, padding=padding)
    expected_h = h if padding == "same" else h - 4
    expected_w = w if padding == "same" else w - 4
    assert y.shape == (b, _expected_out_channels(mode, c, k), expected_h, expected_w)


def test_same_reflect_falls_back_to_replicate_below_3() -> None:
    """A dimension smaller than 3 cannot be reflect-padded by 2; this must
    silently fall back to replicate rather than raising, and match the
    explicit replicate result."""
    x = torch.rand(2, 3, 2, 9)
    reflect = srm_conv2d(x, bank="square3", mode="sum", padding="same", padding_mode="reflect")
    replicate = srm_conv2d(x, bank="square3", mode="sum", padding="same", padding_mode="replicate")
    assert torch.equal(reflect, replicate)


def test_input_scale_makes_truncation_range_independent() -> None:
    torch.manual_seed(0)
    x01 = torch.rand(2, 3, 9, 9)
    x255 = x01 * 255.0
    y01 = srm_conv2d(x01)
    y255 = srm_conv2d(x255, input_scale="0-255")
    assert torch.allclose(y01, y255 / 255.0, atol=1e-6)
    # The threshold is small relative to the raw response, so this is a
    # meaningful check of truncation, not just of the linear conv.
    assert (y255.abs() == 3.0).any()


def test_tlu() -> None:
    x = torch.tensor([-10.0, -3.0, -1.0, 0.0, 1.0, 3.0, 10.0])
    original = x.clone()
    y = tlu(x, 3.0)
    assert torch.equal(y, torch.tensor([-3.0, -3.0, -1.0, 0.0, 1.0, 3.0, 3.0]))
    assert torch.equal(x, original)  # not mutated in place


def test_per_channel_needs_k_equal_c() -> None:
    x = torch.zeros(1, 3, 8, 8)
    with pytest.raises(ValueError, match=r"K == C.*K=30.*C=3"):
        srm_conv2d(x, bank="srm30", mode="per-channel")


def test_gray_needs_three_channels() -> None:
    x = torch.zeros(1, 4, 8, 8)
    with pytest.raises(ValueError, match=r"gray.*C=4"):
        srm_conv2d(x, bank="srm30", mode="gray")


def test_no_in_place_on_input() -> None:
    x = torch.rand(2, 3, 9, 9)
    original = x.clone()
    srm_conv2d(x, bank="rgbn3", mode="sum")
    assert torch.equal(x, original)


# --- argument validation: invalid options must raise, not be silently
# --- misinterpreted as one of the valid choices ------------------------------


def test_invalid_input_scale_raises() -> None:
    x = torch.zeros(1, 3, 8, 8)
    with pytest.raises(ValueError, match=r"input_scale.*'0-1'.*'0-255'"):
        srm_conv2d(x, bank="square3", mode="sum", input_scale="0-100")  # type: ignore[arg-type]


def test_invalid_padding_raises() -> None:
    x = torch.zeros(1, 3, 8, 8)
    with pytest.raises(ValueError, match=r"padding.*'same'.*'valid'"):
        srm_conv2d(x, bank="square3", mode="sum", padding="full")  # type: ignore[arg-type]


def test_invalid_padding_mode_raises() -> None:
    x = torch.zeros(1, 3, 8, 8)
    with pytest.raises(ValueError, match=r"padding_mode.*'reflect'.*'zeros'.*'replicate'"):
        srm_conv2d(x, bank="square3", mode="sum", padding_mode="circular")  # type: ignore[arg-type]


def test_invalid_truncate_raises() -> None:
    x = torch.zeros(1, 3, 8, 8)
    with pytest.raises(ValueError, match=r"truncate.*-1\.0"):
        srm_conv2d(x, bank="square3", mode="sum", truncate=-1.0)


def test_truncate_none_is_still_valid() -> None:
    x = torch.zeros(1, 3, 8, 8)
    srm_conv2d(x, bank="square3", mode="sum", truncate=None)  # no raise


def test_padding_valid_on_too_small_image_raises_value_error() -> None:
    x = torch.zeros(1, 3, 4, 4)
    with pytest.raises(ValueError, match=r"padding='valid'.*H=4.*W=4"):
        srm_conv2d(x, bank="square3", mode="sum", padding="valid")


def test_weight_override_is_not_on_the_public_signature() -> None:
    """`SRMConv2d.forward` shares its implementation with `srm_conv2d` via a
    private helper that takes a pre-resolved kernel tensor; that hook must
    never leak onto the public `srm_conv2d` signature."""
    params = inspect.signature(srm_conv2d).parameters
    assert "_weight" not in params
    assert "weight" not in params


@pytest.mark.parametrize("shape", [(1, 3, 0, 8), (1, 3, 8, 0)])
def test_empty_spatial_dimension_raises_value_error(shape: tuple[int, int, int, int]) -> None:
    """An empty spatial dimension (H=0 or W=0) must raise the package's own
    `ValueError`, like the too-small `padding="valid"` case, rather than a
    confusing `RuntimeError` from deep inside `F.conv2d`/`F.pad`."""
    x = torch.zeros(shape)
    with pytest.raises(ValueError, match=r"H > 0 and W > 0"):
        srm_conv2d(x, bank="square3", mode="sum", padding="same")


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int64])
def test_srm_conv2d_rejects_non_floating_input(dtype: torch.dtype) -> None:
    x = (torch.rand(1, 3, 8, 8) * 255).to(dtype)
    with pytest.raises(TypeError, match=r"floating-point tensor.*x\.float\(\)"):
        srm_conv2d(x, bank="square3", mode="sum")


def test_srm_conv2d_unknown_bank_raises_keyerror() -> None:
    x = torch.zeros(1, 3, 8, 8)
    with pytest.raises(KeyError):
        srm_conv2d(x, bank="nope")


def test_srm_conv2d_non_string_kernel_name_raises_typeerror() -> None:
    x = torch.zeros(1, 3, 8, 8)
    with pytest.raises(TypeError):
        srm_conv2d(x, bank=[1, 2])  # type: ignore[list-item]
