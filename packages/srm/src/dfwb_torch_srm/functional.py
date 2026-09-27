"""Stateless functional API for dfwb_torch_srm: SRM convolution and TLU.

Both functions are pure PyTorch: dtype and device follow the input, neither
mutates its input in place, and both accept batched ``[B, C, H, W]`` input
only. Kernels come from `dfwb_torch_srm.kernels.srm_kernels`.
"""

from collections.abc import Sequence
from typing import Literal

import torch
import torch.nn.functional as F

from dfwb_torch_srm.kernels import srm_kernels

__all__ = ["srm_conv2d", "tlu"]

_KERNEL_SIZE = 5
_SAME_PAD = (_KERNEL_SIZE - 1) // 2  # 2 on every side, for stride-1 "same" output

_INPUT_SCALES: tuple[str, ...] = ("0-1", "0-255")
_PADDINGS: tuple[str, ...] = ("same", "valid")
_PADDING_MODES: tuple[str, ...] = ("reflect", "zeros", "replicate")


def _check_choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    """Raise `ValueError` listing ``allowed`` unless ``value`` is one of them."""
    if value not in allowed:
        options = ", ".join(repr(option) for option in allowed)
        raise ValueError(f"{name}={value!r} is not one of {options}")


def _check_truncate(truncate: float | None) -> None:
    """Raise `ValueError` unless ``truncate`` is `None` or non-negative."""
    if truncate is not None and truncate < 0:
        raise ValueError(f"truncate must be None or >= 0; got {truncate!r}")


def _check_non_negative(name: str, value: float) -> None:
    """Raise `ValueError` unless ``value`` is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be >= 0; got {value!r}")


def _check_srm_conv2d_options(
    *,
    input_scale: str,
    padding: str,
    padding_mode: str,
    truncate: float | None,
) -> None:
    """Validate the options shared by `srm_conv2d` and `SRMConv2d`.

    Raises `ValueError`, naming the offending option and, for the
    string-choice options, every value it accepts, instead of letting an
    invalid option be silently misread as one of the valid choices (or fail
    later with an unrelated, confusing error from `torch.nn.functional`).

    Args:
        input_scale: Must be ``"0-1"`` or ``"0-255"``.
        padding: Must be ``"same"`` or ``"valid"``.
        padding_mode: Must be ``"reflect"``, ``"zeros"`` or ``"replicate"``.
        truncate: Must be `None` or non-negative.

    Raises:
        ValueError: Any argument is not one of its allowed values.
    """
    _check_choice("input_scale", input_scale, _INPUT_SCALES)
    _check_choice("padding", padding, _PADDINGS)
    _check_choice("padding_mode", padding_mode, _PADDING_MODES)
    _check_truncate(truncate)


# Luma weights for standard-definition R'G'B', from the R'G'B' to Y'CbCr
# matrix in ITU-R Recommendation BT.601-7 (03/2011): Y = 0.299 R + 0.587 G
# + 0.114 B.
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)


def tlu(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Truncated linear unit: clamp every element to ``[-threshold, threshold]``.

    Args:
        x: Input tensor of any shape.
        threshold: The clamp bound; must be non-negative for the bounds to
            be ordered, but this is not checked.

    Returns:
        A new tensor, ``x`` clamped to ``[-threshold, threshold]``. ``x`` is
        not modified.

    Example:
        >>> import torch
        >>> tlu(torch.tensor([-5.0, 0.0, 5.0]), 3.0)
        tensor([-3.,  0.,  3.])
    """
    return x.clamp(min=-threshold, max=threshold)


def _same_padding_mode(padding_mode: str, height: int, width: int) -> str:
    # torch's reflect padding needs pad < size on every padded dimension; a
    # "same" pad of 2 needs height, width > 2. Below that, reflection is
    # impossible, so this falls back to replicate (documented in
    # srm_conv2d's docstring).
    if padding_mode == "reflect" and (height < 3 or width < 3):
        return "replicate"
    return padding_mode


def _pad_same(x: torch.Tensor, padding_mode: str, height: int, width: int) -> torch.Tensor:
    mode = _same_padding_mode(padding_mode, height, width)
    if mode == "zeros":
        return F.pad(x, (_SAME_PAD,) * 4, mode="constant", value=0.0)
    return F.pad(x, (_SAME_PAD,) * 4, mode=mode)


def _luminance(x: torch.Tensor) -> torch.Tensor:
    channels = x.shape[1]
    if channels != 3:
        raise ValueError(
            f"srm_conv2d(mode='gray') needs 3 input channels (R, G, B); got C={channels}"
        )
    weights = torch.tensor(_LUMA_WEIGHTS, dtype=x.dtype, device=x.device).view(1, 3, 1, 1)
    return (x * weights).sum(dim=1, keepdim=True)


def _mode_weight(kernels: torch.Tensor, mode: str, channels: int) -> tuple[torch.Tensor, int]:
    """Build the `F.conv2d` weight and groups for `mode`.

    Args:
        kernels: The kernel bank, shape ``[K, 5, 5]``.
        mode: One of ``"depthwise"``, ``"sum"``, ``"gray"``, ``"per-channel"``.
        channels: The number of input channels the weight will be applied to
            (for ``"gray"``, this is 1, the luminance channel).

    Returns:
        The ``(weight, groups)`` pair for `F.conv2d`.

    Raises:
        ValueError: ``mode == "per-channel"`` and ``K != channels``.
    """
    num_kernels = kernels.shape[0]
    if mode in ("gray", "per-channel"):
        if mode == "per-channel" and num_kernels != channels:
            raise ValueError(
                "srm_conv2d(mode='per-channel') needs K == C; "
                f"got K={num_kernels} kernels for C={channels} channels"
            )
        # gray: channels == 1 (the luminance plane), one output per kernel.
        # per-channel: K == C, kernel i on channel i, C outputs.
        return kernels.unsqueeze(1), channels if mode == "per-channel" else 1
    if mode == "sum":
        weight = kernels.unsqueeze(1).expand(num_kernels, channels, _KERNEL_SIZE, _KERNEL_SIZE)
        return weight.contiguous(), 1
    if mode == "depthwise":
        return kernels.unsqueeze(1).repeat(channels, 1, 1, 1), channels
    raise ValueError(
        f"srm_conv2d: unknown mode {mode!r}; expected 'depthwise', 'sum', 'gray' or 'per-channel'"
    )


def srm_conv2d(
    x: torch.Tensor,
    *,
    bank: str | Sequence[str] = "srm30",
    mode: Literal["depthwise", "sum", "gray", "per-channel"] = "depthwise",
    truncate: float | None = 3.0,
    input_scale: Literal["0-1", "0-255"] = "0-1",
    padding: Literal["same", "valid"] = "same",
    padding_mode: Literal["reflect", "zeros", "replicate"] = "reflect",
    _weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convolve a batched image with an SRM/high-pass kernel bank, then truncate.

    The input is ``[B, C, H, W]``. Kernels come from
    `dfwb_torch_srm.kernels.srm_kernels`; dtype and device follow ``x``.

    Modes:
        * ``"depthwise"``: every kernel is applied to every channel
          independently, giving ``C * K`` output channels, ordered
          channel-major (``c0k0, c0k1, ..., c0k{K-1}, c1k0, ...``).
        * ``"sum"``: every kernel is applied to every channel and the C
          results are summed, giving ``K`` output channels.
        * ``"gray"``: the input is first reduced to luminance (ITU-R BT.601:
          ``0.299 R + 0.587 G + 0.114 B``), which needs ``C == 3``, then
          every kernel is applied to that single plane, giving ``K`` output
          channels.
        * ``"per-channel"``: kernel ``i`` is applied to channel ``i``, giving
          ``C`` output channels. This needs ``K == C``; otherwise a
          `ValueError` names both.

    Truncation (`tlu`) is expressed in 0-255 residual units, as in the SRM
    literature. With ``input_scale="0-1"``, the effective threshold is
    ``truncate / 255`` so that scaling the whole input by 255 and passing
    ``input_scale="0-255"`` gives the same (rescaled) result.

    ``padding="same"`` pads by 2 on every side (kernels are always 5x5) with
    ``padding_mode``. Reflect padding needs the padded dimension to exceed
    the pad amount, i.e. ``H > 2`` and ``W > 2``; when ``H`` or ``W`` is
    smaller than 3, reflect is impossible and this silently falls back to
    replicate padding instead.

    Args:
        x: Input, shape ``[B, C, H, W]``.
        bank: A bank name from `dfwb_torch_srm.kernels.BANKS`, or a sequence
            of kernel names, giving ``K`` kernels.
        mode: One of ``"depthwise"``, ``"sum"``, ``"gray"``, ``"per-channel"``.
        truncate: The TLU threshold in 0-255 units, or `None` to disable
            truncation.
        input_scale: ``"0-1"`` or ``"0-255"``, the scale ``x`` is in. Only
            affects the truncation threshold.
        padding: ``"same"`` (output has the same ``H, W`` as ``x``) or
            ``"valid"`` (no padding; ``H, W`` shrink by 4).
        padding_mode: ``"reflect"``, ``"zeros"`` or ``"replicate"``, used
            only when ``padding="same"``.
        _weight: Module-internal only, not part of the public API: a
            ``[K, 5, 5]`` kernel tensor to use in place of resolving
            ``bank`` via `srm_kernels`. `dfwb_torch_srm.modules.SRMConv2d`
            passes its current (fixed or trainable) weight through here so
            its forward pass shares this exact function; ``bank`` is
            ignored when this is given. It is cast to ``x``'s dtype and
            device like the resolved bank would be.

    Returns:
        The filtered, truncated output. Its channel count depends on
        ``mode``; its dtype and device match ``x``.

    Raises:
        ValueError: ``mode="gray"`` and ``C != 3``; ``mode="per-channel"``
            and ``K != C``; an unrecognised ``mode``; ``input_scale``,
            ``padding`` or ``padding_mode`` not one of their allowed values;
            ``truncate`` negative; or ``padding="valid"`` on an image
            smaller than 5x5 (a 5x5 kernel does not fit with no padding).

    Example:
        >>> import torch
        >>> x = torch.zeros(1, 3, 8, 8)
        >>> srm_conv2d(x, bank="square3", mode="sum").shape
        torch.Size([1, 1, 8, 8])
        >>> srm_conv2d(x, bank="square3", mode="sum").abs().max().item()
        0.0
    """
    _check_srm_conv2d_options(
        input_scale=input_scale, padding=padding, padding_mode=padding_mode, truncate=truncate
    )
    if x.ndim != 4:
        raise ValueError(f"srm_conv2d expects a [B, C, H, W] tensor; got shape {tuple(x.shape)}")
    _, _, height, width = x.shape
    if padding == "valid" and (height < _KERNEL_SIZE or width < _KERNEL_SIZE):
        raise ValueError(
            f"srm_conv2d(padding='valid') needs an image at least {_KERNEL_SIZE}x{_KERNEL_SIZE} "
            f"(the kernel size), since there is no padding to fit a smaller one; "
            f"got H={height}, W={width}"
        )

    if _weight is not None:
        kernels = _weight.to(dtype=x.dtype, device=x.device)  # [K, 5, 5]
    else:
        kernels = srm_kernels(bank, dtype=x.dtype, device=x.device)  # [K, 5, 5]

    plane = _luminance(x) if mode == "gray" else x
    weight, groups = _mode_weight(kernels, mode, plane.shape[1])

    if padding == "same":
        plane = _pad_same(plane, padding_mode, height, width)
    y = F.conv2d(plane, weight, groups=groups)

    if truncate is not None:
        threshold = truncate if input_scale == "0-255" else truncate / 255.0
        y = tlu(y, threshold)
    return y
