"""nn.Module wrappers for dfwb_torch_srm: `SRMConv2d` and `TLU`.

Both modules are thin `nn.Module` wrappers over the stateless functional API
(`dfwb_torch_srm.functional`); neither adds any global state, and both
follow the input's device and dtype.
"""

from collections.abc import Sequence
from typing import Literal

import torch
from torch import nn
from torch.nn.utils import parametrize

from dfwb_torch_srm.functional import (
    _INPUT_SCALES,
    _check_choice,
    _check_positive,
    _check_srm_conv2d_options,
    _srm_conv2d,
    tlu,
)
from dfwb_torch_srm.kernels import srm_kernels

__all__ = ["TLU", "SRMConv2d"]


def _out_channels(mode: str, in_channels: int, num_kernels: int) -> int:
    """The output channel count for `SRMConv2d`, given its mode.

    Args:
        mode: One of ``"depthwise"``, ``"sum"``, ``"gray"``, ``"per-channel"``.
        in_channels: The module's ``in_channels``.
        num_kernels: ``K``, the number of kernels in the module's bank.

    Returns:
        ``in_channels * num_kernels`` for ``"depthwise"``; ``num_kernels``
        for ``"sum"`` or ``"gray"``; ``in_channels`` (which then equals
        ``num_kernels``) for ``"per-channel"``.

    Raises:
        ValueError: ``mode == "per-channel"`` and ``num_kernels != in_channels``,
            ``mode == "gray"`` and ``in_channels != 3``, or an unrecognised
            ``mode``.
    """
    if mode == "depthwise":
        return in_channels * num_kernels
    if mode == "gray":
        if in_channels != 3:
            raise ValueError(
                f"SRMConv2d(mode='gray') needs 3 input channels (R, G, B); got C={in_channels}"
            )
        return num_kernels
    if mode == "sum":
        return num_kernels
    if mode == "per-channel":
        if num_kernels != in_channels:
            raise ValueError(
                "SRMConv2d(mode='per-channel') needs K == C; "
                f"got K={num_kernels} kernels for C={in_channels} channels"
            )
        return in_channels
    raise ValueError(
        f"SRMConv2d: unknown mode {mode!r}; expected 'depthwise', 'sum', 'gray' or 'per-channel'"
    )


class _ZeroMeanPerKernel(nn.Module):
    """Parametrisation: re-centre every ``[5, 5]`` kernel to zero mean.

    Registered on `SRMConv2d.weight` with
    `torch.nn.utils.parametrize.register_parametrization` when
    ``trainable=True``. `torch.nn.utils.parametrize` recomputes this on
    every access to `.weight` (so on every forward pass and every gradient
    step), not just once at construction, so an optimiser can never drift
    the kernels' DC gain away from 0.

    Example:
        >>> import torch
        >>> _ = torch.manual_seed(0)
        >>> w = torch.rand(2, 5, 5, dtype=torch.float64)
        >>> _ZeroMeanPerKernel()(w).sum(dim=(-2, -1)).abs().max().item() < 1e-12
        True
    """

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """Subtract each kernel's own mean.

        Args:
            weight: Kernel weights, shape ``[K, 5, 5]``.

        Returns:
            ``weight`` with every ``[5, 5]`` kernel's mean subtracted, so
            each one sums to (numerically) zero.
        """
        return weight - weight.mean(dim=(-2, -1), keepdim=True)


class SRMConv2d(nn.Module):
    """SRM / high-pass residual convolution, as an `nn.Module`.

    A thin wrapper around `dfwb_torch_srm.functional.srm_conv2d`: see that
    function's docstring for the exact semantics of ``mode``, ``truncate``,
    ``input_scale``, ``padding`` and ``padding_mode``. This module adds the
    weight itself (a buffer or a parameter) and `out_channels`.

    Fixed kernels (``trainable=False``, the default) are a non-trainable
    `register_buffer`, so they never appear in `.parameters()` and are never
    touched by an optimiser. The buffer is built once at construction,
    **in float64**, from `bank`'s exact taps -- regardless of
    ``torch.get_default_dtype()`` -- so `forward` always casts down from
    the exact values on every call, never re-rounding through some other,
    lower, intermediate dtype first. With ``trainable=True``, `weight` is
    instead an `nn.Parameter`, built at the default dtype (unchanged by the
    above: a trained parameter has no more-exact source to rebuild from),
    re-centred to zero mean per kernel on every access by a
    `torch.nn.utils.parametrize` parametrisation
    (``w - w.mean(dim=(-2, -1), keepdim=True)``), so gradient steps can
    never turn a residual kernel into a low-pass filter: its DC gain stays
    exactly 0.

    `forward` is a pure function of `weight` and its input: on every call,
    it casts the *current* `weight` to the input's dtype and device (so the
    layer follows the input, with no silent upcast, and no state is cached
    across calls), then calls `srm_conv2d`. `weight` is the sole source of
    truth: editing it in place, or loading a different one with
    `load_state_dict` (including one saved by an older release whose buffer
    was a different dtype -- the copy casts to this buffer's own dtype),
    changes what `forward` computes from the very next call. Being stateless
    this way is what makes the module work under `torch.inference_mode`,
    `torch.jit.trace`, `torch.compile` and `torch.export`: nothing is
    mutated on a "warm-up" call that a later, differently-traced or
    differently-moded call could then see.

    Attributes:
        out_channels: The number of output channels, fixed at construction
            from ``mode``, the number of kernels ``K`` in ``bank``, and
            ``in_channels``: ``in_channels * K`` for ``"depthwise"``, ``K``
            for ``"sum"``/``"gray"``, and ``in_channels`` (== ``K``) for
            ``"per-channel"``.

    Example:
        >>> import torch
        >>> layer = SRMConv2d(in_channels=3, bank="rgbn3", mode="sum")
        >>> layer.out_channels
        3
        >>> layer(torch.zeros(1, 3, 8, 8)).shape
        torch.Size([1, 3, 8, 8])
    """

    in_channels: int
    bank: str | Sequence[str]
    mode: Literal["depthwise", "sum", "gray", "per-channel"]
    trainable: bool
    truncate: float | None
    input_scale: Literal["0-1", "0-255"]
    padding: Literal["same", "valid"]
    padding_mode: Literal["reflect", "zeros", "replicate"]
    out_channels: int
    weight: torch.Tensor

    def __init__(
        self,
        in_channels: int = 3,
        *,
        bank: str | Sequence[str] = "srm30",
        mode: Literal["depthwise", "sum", "gray", "per-channel"] = "depthwise",
        trainable: bool = False,
        truncate: float | None = 3.0,
        input_scale: Literal["0-1", "0-255"] = "0-1",
        padding: Literal["same", "valid"] = "same",
        padding_mode: Literal["reflect", "zeros", "replicate"] = "reflect",
    ) -> None:
        """Build the layer.

        Args:
            in_channels: The number of input channels. Must be >= 1.
            bank: A bank name from `dfwb_torch_srm.kernels.BANKS`, or a
                sequence of kernel names, giving ``K`` kernels.
            mode: One of ``"depthwise"``, ``"sum"``, ``"gray"``, ``"per-channel"``.
            trainable: If `True`, `weight` is a learnable, zero-DC-constrained
                `nn.Parameter`, built at the default dtype; if `False` (the
                default), a fixed buffer, built in float64 regardless of the
                default dtype (see the class docstring).
            truncate: The TLU threshold in 0-255 units, or `None` to disable
                truncation.
            input_scale: ``"0-1"`` or ``"0-255"``, the scale the input is in.
            padding: ``"same"`` or ``"valid"``.
            padding_mode: ``"reflect"``, ``"zeros"`` or ``"replicate"``, used
                only when ``padding="same"``.

        Raises:
            ValueError: ``in_channels < 1``; ``mode="per-channel"`` and
                ``K != in_channels``; ``mode="gray"`` and
                ``in_channels != 3``; an unrecognised ``mode``; or
                ``input_scale``, ``padding``, ``padding_mode`` or
                ``truncate`` not one of their allowed values.
            KeyError: ``bank`` (or a name inside it) is not a known
                bank/kernel name in
                `dfwb_torch_srm.kernels.BANKS`/`KERNELS`.
            TypeError: ``bank`` is a sequence containing a non-string name.
        """
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"SRMConv2d needs in_channels >= 1; got {in_channels!r}")
        _check_srm_conv2d_options(
            input_scale=input_scale, padding=padding, padding_mode=padding_mode, truncate=truncate
        )
        # Fixed kernels are built in float64, straight from bank's exact
        # taps, regardless of torch.get_default_dtype(): forward always
        # casts *down* from this exact source on every call, so it never
        # re-rounds through some other, lower, dtype first (see the class
        # docstring). A trainable parameter has no such "exact" source to
        # protect -- it is whatever training makes it -- so it keeps using
        # the default dtype, as before.
        kernels = srm_kernels(bank) if trainable else srm_kernels(bank, dtype=torch.float64)
        self.in_channels = in_channels
        self.bank = bank
        self.mode = mode
        self.trainable = trainable
        self.truncate = truncate
        self.input_scale = input_scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.out_channels = _out_channels(mode, in_channels, kernels.shape[0])

        if trainable:
            self.weight = nn.Parameter(kernels)
            parametrize.register_parametrization(self, "weight", _ZeroMeanPerKernel())
        else:
            self.register_buffer("weight", kernels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the SRM/high-pass convolution, then truncate.

        A pure function of `weight` (see the class docstring) and ``x``:
        mutates no module state, so a call under `torch.inference_mode`, a
        `torch.jit.trace`/`torch.export` capture, or any other special
        context leaves nothing behind that could affect a later, ordinary
        call.

        Args:
            x: Input, shape ``[B, in_channels, H, W]``.

        Returns:
            The filtered, truncated output, shape
            ``[B, out_channels, H', W']`` (``H'``, ``W'`` per ``padding``, as
            in `srm_conv2d`); dtype and device match ``x``.

        Raises:
            TypeError: ``x`` is not a floating-point tensor (call
                ``x.float()`` first).
            ValueError: ``x``'s channel count does not match `in_channels`
                (checked here since `mode="depthwise"`/``"sum"`` would
                otherwise not notice: their weight only depends on ``x``'s
                actual channel count, not on `in_channels`, so a mismatch
                would silently give an output whose channel count no longer
                matches `out_channels`, instead of raising); or ``x`` has an
                empty spatial dimension (``H == 0`` or ``W == 0``).
        """
        if x.ndim == 4 and x.shape[1] != self.in_channels:
            raise ValueError(
                f"SRMConv2d(in_channels={self.in_channels}) expects input with "
                f"C={self.in_channels}; got shape {tuple(x.shape)} with C={x.shape[1]}"
            )
        weight = self.weight.to(dtype=x.dtype, device=x.device)
        return _srm_conv2d(
            x,
            mode=self.mode,
            truncate=self.truncate,
            input_scale=self.input_scale,
            padding=self.padding,
            padding_mode=self.padding_mode,
            weight=weight,
        )

    def extra_repr(self) -> str:
        """A one-line summary for `repr(module)` / `print(module)`."""
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"bank={self.bank!r}, mode={self.mode!r}, trainable={self.trainable}, "
            f"truncate={self.truncate}, input_scale={self.input_scale!r}, "
            f"padding={self.padding!r}, padding_mode={self.padding_mode!r}"
        )


class TLU(nn.Module):
    """Truncated linear unit, as an `nn.Module`.

    Clamps its input to ``[-t, t]``, where ``t`` is ``threshold`` (always in
    0-255 residual units, as in the SRM literature, regardless of
    ``input_scale``) or ``threshold / 255`` when ``input_scale="0-1"``. A
    thin wrapper around `dfwb_torch_srm.functional.tlu`, whose own
    ``threshold`` is instead in whatever units its input already is, with
    no such scaling (or validation) -- see its docstring; holds no
    learnable state.

    Example:
        >>> import torch
        >>> module = TLU(threshold=3.0, input_scale="0-255")
        >>> module(torch.tensor([-10.0, 0.0, 10.0]))
        tensor([-3.,  0.,  3.])
    """

    threshold: float
    input_scale: Literal["0-1", "0-255"]

    def __init__(self, threshold: float, input_scale: Literal["0-1", "0-255"] = "0-1") -> None:
        """Build the layer.

        Args:
            threshold: The clamp bound, in 0-255 residual units (not the
                units of the eventual input -- see `dfwb_torch_srm.functional.tlu`
                for the raw, unscaled primitive). Must be strictly positive.
            input_scale: ``"0-1"`` (the default) or ``"0-255"``, the scale
                the input is in. ``"0-1"`` uses an effective threshold of
                ``threshold / 255``, so the same clamp applies whether the
                input is scaled to 0-1 or left at 0-255.

        Raises:
            ValueError: ``input_scale`` is not ``"0-1"`` or ``"0-255"``, or
                ``threshold`` is not strictly positive.
        """
        super().__init__()
        _check_choice("input_scale", input_scale, _INPUT_SCALES)
        _check_positive("threshold", threshold)
        self.threshold = threshold
        self.input_scale = input_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp ``x`` to ``[-t, t]``.

        Args:
            x: Input tensor of any shape.

        Returns:
            A new tensor, ``x`` clamped; ``x`` is not modified.
        """
        t = self.threshold if self.input_scale == "0-255" else self.threshold / 255.0
        return tlu(x, t)

    def extra_repr(self) -> str:
        """A one-line summary for `repr(module)` / `print(module)`."""
        return f"threshold={self.threshold}, input_scale={self.input_scale!r}"
