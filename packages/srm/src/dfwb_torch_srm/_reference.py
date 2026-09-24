"""Numpy/scipy reference implementation of `srm_conv2d`, for tests only.

`tests/test_reference.py` checks the PyTorch implementation
(`dfwb_torch_srm.functional.srm_conv2d`) against this independent
implementation, built on `scipy.signal.correlate2d` (no kernel flip, matching
the cross-correlation convention `kernels.py` documents) with padding from
`numpy.pad`. numpy and scipy are dev-only dependencies (`pyproject.toml`'s
`[dependency-groups] dev`); nothing under `src/` other than this file imports
them, and `functional.py` does not import numpy or scipy at all.
"""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from scipy.signal import correlate2d  # type: ignore[import-untyped]

from dfwb_torch_srm.kernels import srm_kernels

# Kept identical to functional.py's _LUMA_WEIGHTS (ITU-R BT.601, as given in
# the package plan and the M6 task brief).
_LUMA_WEIGHTS = (0.299, 0.587, 0.114)


def _padding_mode_for_numpy(padding_mode: str, height: int, width: int) -> str:
    if padding_mode == "reflect" and (height < 3 or width < 3):
        return "edge"  # numpy.pad's name for replicate/edge padding
    if padding_mode == "replicate":
        return "edge"
    if padding_mode == "zeros":
        return "constant"
    return padding_mode


def _correlate(plane: np.ndarray, kernel: np.ndarray, padding: str, np_mode: str) -> np.ndarray:
    if padding == "same":
        plane = np.pad(plane, 2, mode=np_mode)  # type: ignore[call-overload]
    result: np.ndarray = correlate2d(plane, kernel, mode="valid")
    return result


def srm_conv2d_reference(
    x: np.ndarray,
    *,
    bank: str | Sequence[str] = "srm30",
    mode: Literal["depthwise", "sum", "gray", "per-channel"] = "depthwise",
    truncate: float | None = 3.0,
    input_scale: Literal["0-1", "0-255"] = "0-1",
    padding: Literal["same", "valid"] = "same",
    padding_mode: Literal["reflect", "zeros", "replicate"] = "reflect",
) -> np.ndarray:
    """A float64 numpy reference for `srm_conv2d`.

    Loops explicitly over batch, channel and kernel; not meant to be fast,
    only independent of the PyTorch implementation under test.

    Args:
        x: Input, shape ``[B, C, H, W]``, any real numpy dtype (computed as
            float64 internally).
        bank: Same as `srm_conv2d`.
        mode: Same as `srm_conv2d`.
        truncate: Same as `srm_conv2d`.
        input_scale: Same as `srm_conv2d`.
        padding: Same as `srm_conv2d`.
        padding_mode: Same as `srm_conv2d`.

    Returns:
        The filtered, truncated output as a float64 array.

    Raises:
        ValueError: ``mode="gray"`` and ``C != 3``; ``mode="per-channel"``
            and ``K != C``.
    """
    x = np.asarray(x, dtype=np.float64)
    b, c, h, w = x.shape
    kernels = srm_kernels(bank, dtype=torch.float64).numpy()  # [K, 5, 5]
    k = kernels.shape[0]

    if mode == "gray":
        if c != 3:
            raise ValueError(f"srm_conv2d(mode='gray') needs 3 input channels (R, G, B); got C={c}")
        weights = np.array(_LUMA_WEIGHTS).reshape(1, 3, 1, 1)
        x = (x * weights).sum(axis=1, keepdims=True)
        c = 1
    elif mode == "per-channel" and k != c:
        raise ValueError(
            f"srm_conv2d(mode='per-channel') needs K == C; got K={k} kernels for C={c} channels"
        )

    np_mode = _padding_mode_for_numpy(padding_mode, h, w)
    out_h = h if padding == "same" else h - 4
    out_w = w if padding == "same" else w - 4

    if mode == "depthwise":
        out = np.empty((b, c * k, out_h, out_w))
        for bi in range(b):
            for ci in range(c):
                for ki in range(k):
                    out[bi, ci * k + ki] = _correlate(x[bi, ci], kernels[ki], padding, np_mode)
    elif mode == "sum":
        out = np.zeros((b, k, out_h, out_w))
        for bi in range(b):
            for ki in range(k):
                for ci in range(c):
                    out[bi, ki] += _correlate(x[bi, ci], kernels[ki], padding, np_mode)
    elif mode == "gray":
        out = np.empty((b, k, out_h, out_w))
        for bi in range(b):
            for ki in range(k):
                out[bi, ki] = _correlate(x[bi, 0], kernels[ki], padding, np_mode)
    elif mode == "per-channel":
        out = np.empty((b, c, out_h, out_w))
        for bi in range(b):
            for ci in range(c):
                out[bi, ci] = _correlate(x[bi, ci], kernels[ci], padding, np_mode)
    else:
        raise ValueError(
            f"srm_conv2d: unknown mode {mode!r}; "
            "expected 'depthwise', 'sum', 'gray' or 'per-channel'"
        )

    if truncate is not None:
        threshold = truncate if input_scale == "0-255" else truncate / 255.0
        out = np.clip(out, -threshold, threshold)
    return out
