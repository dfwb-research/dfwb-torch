"""Canonical SRM, RGB-N and textbook high-pass kernel banks.

Named, cited, zero-sum 5x5 residual kernels: the 30 base residual filters of
the spatial rich model (SRM), the three kernels RGB-N feeds its noise stream,
and textbook Laplacian and Sobel kernels for ablations.

Conventions:
    Every kernel is a 5x5 grid of cross-correlation taps, the convention of
    ``torch.nn.functional.conv2d``: the output at pixel ``(i, j)`` is
    ``sum(taps[r][c] * X[i + r - 2, j + c - 2])``, so ``taps[2][2]``
    multiplies the centre pixel. 3x3 kernels are zero-embedded in the centre.
    Orientation suffixes:

    * ``-h`` runs along the row and ``-v`` along the column (its transpose);
    * ``-d1`` runs along the main diagonal (towards the bottom right) and
      ``-d2`` along the anti-diagonal (towards the bottom left, the
      left-right mirror of ``-d1``);
    * ``-rev`` is the same kernel rotated by 180 degrees: the opposite
      direction for a first- or third-order residual, the opposite half for
      an EDGE kernel.

    An EDGE ``-h`` kernel is the half above the centre row, as E3a is drawn
    in Fridrich & Kodovsky 2012 Fig. 2 (SRM.m's ``Du``); ``-v`` is the left
    half (``Dl``). ``docs/kernels.md`` lists every kernel with its SRM.m name.

Normalisation:
    SRM writes each residual as ``R = Xhat(N) - c * X`` with ``c`` the centre
    coefficient (Fridrich & Kodovsky 2012, Sec. II-A, Eq. (1)). Every
    ``srm30`` kernel is divided by ``c`` (1, 2, 3, 4 or 12), so its centre
    tap is -1 and residual magnitudes are comparable across orders. This is
    SRM's own smallest quantisation step, ``q = c``: Eq. (2) quantises
    ``R / q`` and Sec. II-D, Eq. (9) takes ``q`` in ``{c, 1.5c, 2c}``; the
    authors' SRM.m (lines 39-44) calls its extractors with ``q * c`` for
    ``q = 1, 1.5, 2`` and notes "Recommended values of q are c, 1.5c, 2c,
    where c is the central coefficient". The ``rgbn3`` kernels are RGB-N's
    exactly (SQUARE 5x5 / 12, SQUARE 3x3 / 4, second-order horizontal / 2, in
    the authors' code order), which are the same ``srm30`` kernels. The
    Laplacian and Sobel kernels keep their textbook integer coefficients.

Sources:
    * ``srm30``, ``square3``, ``square5``: J. Fridrich and J. Kodovsky, "Rich
      Models for Steganalysis of Digital Images", IEEE TIFS 7(3):868-882,
      2012, doi:10.1109/TIFS.2012.2190402: Sec. II-A Eq. (1)-(2), Sec. II-B
      and Fig. 2, Sec. II-D Eq. (9). The kernels were transcribed from the
      authors' MATLAB extractor, SRM.m in
      http://dde.binghamton.edu/download/feature_extractors/download/SRM.zip
      (the site has no version control; SRM.m sha256
      b11272825ffb9c296505cda1888dc6702c21c9163b93cf8c3fe67c3854c1aa3e):
      ``all1st``/``all3rd`` for the eight first- and third-order directions,
      ``Residual`` for the second order, SQUARE ('KB', 'KV') and EDGE 3x3
      ('edge-h', 'edge-v'), and ``all5x5`` for EDGE 5x5. The count of 30
      (8 + 4 + 8 + 1 + 4 + 1 + 4) is the set used by J. Ye, J. Ni and Y. Yi,
      "Deep Learning Hierarchical Representations for Image Steganalysis",
      IEEE TIFS 12(11):2545-2557, 2017, doi:10.1109/TIFS.2017.2710946.
    * ``rgbn3``: P. Zhou, X. Han, V. I. Morariu and L. S. Davis, "Learning
      Rich Features for Image Manipulation Detection", CVPR 2018,
      pp. 1053-1061, doi:10.1109/CVPR.2018.00116, Sec. 3.2 and Fig. 4; and
      the authors' code, https://github.com/pengzhou1108/RGB-N at commit
      fdc612d781a83c1783e46ab1e76bb1cc6aa0f458, lib/nets/resnet_fusion.py
      lines 176-194.
    * ``laplacian``: the 4-neighbour discrete Laplacian, R. C. Gonzalez and
      R. E. Woods, Digital Image Processing, 3rd ed., Pearson, 2008,
      Sec. 3.6.2.
    * ``sobel``: I. Sobel and G. Feldman, "A 3x3 Isotropic Gradient Operator
      for Image Processing", Stanford Artificial Intelligence Project, 1968.

    ``docs/kernels.md`` records where these sources disagree and what was
    chosen.

Example:
    >>> from dfwb_torch_srm.kernels import BANKS, KERNELS, srm_kernels
    >>> len(BANKS["srm30"])
    30
    >>> KERNELS["second-order-h"].taps[2]
    (0.0, 0.5, -1.0, 0.5, 0.0)
    >>> srm_kernels("rgbn3").shape
    torch.Size([3, 5, 5])
"""

import difflib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import torch

__all__ = ["BANKS", "KERNELS", "KernelInfo", "srm_kernels"]

_SIZE = 5
_CENTRE = 2

_Taps = tuple[tuple[float, ...], ...]
_Grid = tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class KernelInfo:
    """One named kernel: its taps, where they come from and how they are scaled.

    Attributes:
        name: The lower-kebab-case key in `KERNELS`, e.g. ``"edge3-h"``.
        taps: The 5x5 cross-correlation taps, row-major; 3x3 kernels are
            zero-embedded in the centre. For the output at pixel ``(i, j)``,
            ``taps[r][c]`` multiplies ``X[i + r - 2, j + c - 2]``.
        source: The citation the taps were transcribed from.
        normalisation: How the source's integer coefficients were scaled.

    Example:
        >>> from dfwb_torch_srm.kernels import KERNELS
        >>> info = KERNELS["first-order-h"]
        >>> info.taps[2]
        (0.0, 0.0, -1.0, 1.0, 0.0)
    """

    name: str
    taps: _Taps
    source: str
    normalisation: str


# --- citations -----------------------------------------------------------------

_FK12 = (
    "Fridrich & Kodovsky 2012, Rich Models for Steganalysis of Digital Images, "
    "IEEE TIFS 7(3):868-882, doi:10.1109/TIFS.2012.2190402"
)
_SRM_M = "SRM.m (dde.binghamton.edu/download/feature_extractors, SRM.zip)"
_ZHMD18 = (
    "RGB-N kernel: Zhou, Han, Morariu & Davis 2018, Learning Rich Features for Image "
    "Manipulation Detection, CVPR, doi:10.1109/CVPR.2018.00116, Sec. 3.2 and Fig. 4; "
    "github.com/pengzhou1108/RGB-N@fdc612d781a83c1783e46ab1e76bb1cc6aa0f458 "
    "lib/nets/resnet_fusion.py lines 176-194"
)
_GW08 = (
    "4-neighbour discrete Laplacian: Gonzalez & Woods, Digital Image Processing, "
    "3rd ed., Pearson, 2008, Sec. 3.6.2"
)
_SOBEL68 = (
    "Sobel & Feldman 1968, A 3x3 Isotropic Gradient Operator for Image Processing, "
    "Stanford Artificial Intelligence Project"
)
_TEXTBOOK_NORMALISATION = "none: the textbook integer coefficients"


def _srm_normalisation(c: int) -> str:
    return (
        f"divided by {c}, the magnitude of the centre coefficient c "
        f"(Fridrich & Kodovsky 2012 Eq. (1)); the smallest SRM quantisation step "
        f"q = c (Eq. (9); SRM.m lines 39-44), so the centre tap is -1"
    )


# --- transcriptions ------------------------------------------------------------
# First-, second- and third-order residuals as {(row offset, column offset):
# coefficient}, term by term from the SRM.m expressions quoted alongside
# (X(I+di, J+dj) is the pixel at offset (di, dj) from the centre X(I, J)).

_Offsets = Mapping[tuple[int, int], int]

# (name, SRM.m expression, offsets). SRM.m all1st, c = 1; Fig. 2 (1a-1h).
_FIRST_ORDER: tuple[tuple[str, str, _Offsets], ...] = (
    ("first-order-h", "R = X(I,J+1)-X(I,J)", {(0, 1): 1, (0, 0): -1}),
    ("first-order-h-rev", "L = X(I,J-1)-X(I,J)", {(0, -1): 1, (0, 0): -1}),
    ("first-order-v", "D = X(I+1,J)-X(I,J)", {(1, 0): 1, (0, 0): -1}),
    ("first-order-v-rev", "U = X(I-1,J)-X(I,J)", {(-1, 0): 1, (0, 0): -1}),
    ("first-order-d1", "RD = X(I+1,J+1)-X(I,J)", {(1, 1): 1, (0, 0): -1}),
    ("first-order-d1-rev", "LU = X(I-1,J-1)-X(I,J)", {(-1, -1): 1, (0, 0): -1}),
    ("first-order-d2", "LD = X(I+1,J-1)-X(I,J)", {(1, -1): 1, (0, 0): -1}),
    ("first-order-d2-rev", "RU = X(I-1,J+1)-X(I,J)", {(-1, 1): 1, (0, 0): -1}),
)

# SRM.m Residual(X, 2, type), c = 2; Fig. 2 (2a-2e).
_SECOND_ORDER: tuple[tuple[str, str, _Offsets], ...] = (
    ("second-order-h", "'hor': X(I,J-1)-2*X(I,J)+X(I,J+1)", {(0, -1): 1, (0, 0): -2, (0, 1): 1}),
    ("second-order-v", "'ver': X(I-1,J)-2*X(I,J)+X(I+1,J)", {(-1, 0): 1, (0, 0): -2, (1, 0): 1}),
    (
        "second-order-d1",
        "'diag': X(I-1,J-1)-2*X(I,J)+X(I+1,J+1)",
        {(-1, -1): 1, (0, 0): -2, (1, 1): 1},
    ),
    (
        "second-order-d2",
        "'mdiag': X(I-1,J+1)-2*X(I,J)+X(I+1,J-1)",
        {(-1, 1): 1, (0, 0): -2, (1, -1): 1},
    ),
)

# SRM.m all3rd, c = 3; Sec. II-B ("replacing X_{i,j+1} - X_ij with
# -X_{i,j+2} + 3X_{i,j+1} - 3X_ij + X_{i,j-1}").
_THIRD_ORDER: tuple[tuple[str, str, _Offsets], ...] = (
    (
        "third-order-h",
        "R = -X(I,J+2)+3*X(I,J+1)-3*X(I,J)+X(I,J-1)",
        {(0, 2): -1, (0, 1): 3, (0, 0): -3, (0, -1): 1},
    ),
    (
        "third-order-h-rev",
        "L = -X(I,J-2)+3*X(I,J-1)-3*X(I,J)+X(I,J+1)",
        {(0, -2): -1, (0, -1): 3, (0, 0): -3, (0, 1): 1},
    ),
    (
        "third-order-v",
        "D = -X(I+2,J)+3*X(I+1,J)-3*X(I,J)+X(I-1,J)",
        {(2, 0): -1, (1, 0): 3, (0, 0): -3, (-1, 0): 1},
    ),
    (
        "third-order-v-rev",
        "U = -X(I-2,J)+3*X(I-1,J)-3*X(I,J)+X(I+1,J)",
        {(-2, 0): -1, (-1, 0): 3, (0, 0): -3, (1, 0): 1},
    ),
    (
        "third-order-d1",
        "RD = -X(I+2,J+2)+3*X(I+1,J+1)-3*X(I,J)+X(I-1,J-1)",
        {(2, 2): -1, (1, 1): 3, (0, 0): -3, (-1, -1): 1},
    ),
    (
        "third-order-d1-rev",
        "LU = -X(I-2,J-2)+3*X(I-1,J-1)-3*X(I,J)+X(I+1,J+1)",
        {(-2, -2): -1, (-1, -1): 3, (0, 0): -3, (1, 1): 1},
    ),
    (
        "third-order-d2",
        "LD = -X(I+2,J-2)+3*X(I+1,J-1)-3*X(I,J)+X(I-1,J+1)",
        {(2, -2): -1, (1, -1): 3, (0, 0): -3, (-1, 1): 1},
    ),
    (
        "third-order-d2-rev",
        "RU = -X(I-2,J+2)+3*X(I-1,J+1)-3*X(I,J)+X(I+1,J-1)",
        {(-2, 2): -1, (-1, 1): 3, (0, 0): -3, (1, -1): 1},
    ),
)

# SQUARE 3x3 (S3a), c = 4: Fig. 2 (S3a); SRM.m Residual 'KB'.
_S3A: _Grid = (
    (-1, 2, -1),
    (2, -4, 2),
    (-1, 2, -1),
)

# EDGE 3x3, c = 4: E3a (Fig. 2) in its four rotations; SRM.m Residual
# 'edge-h' (Du, Db) and 'edge-v' (Dl, Dr).
_EDGE3: tuple[tuple[str, str, _Grid], ...] = (
    ("edge3-h", "Du", ((-1, 2, -1), (2, -4, 2), (0, 0, 0))),
    ("edge3-h-rev", "Db", ((0, 0, 0), (2, -4, 2), (-1, 2, -1))),
    ("edge3-v", "Dl", ((-1, 2, 0), (2, -4, 0), (-1, 2, 0))),
    ("edge3-v-rev", "Dr", ((0, 2, -1), (0, -4, 2), (0, 2, -1))),
)

# SQUARE 5x5 (S5a), c = 12: Fig. 2 (S5a); SRM.m Residual 'KV'.
_S5A: _Grid = (
    (-1, 2, -2, 2, -1),
    (2, -6, 8, -6, 2),
    (-2, 8, -12, 8, -2),
    (2, -6, 8, -6, 2),
    (-1, 2, -2, 2, -1),
)

# EDGE 5x5, c = 12: the halves of S5a (Fig. 2 caption; not drawn), from
# SRM.m all5x5 Du, Db, Dl and Dr.
_EDGE5: tuple[tuple[str, str, _Grid], ...] = (
    (
        "edge5-h",
        "Du",
        (
            (-1, 2, -2, 2, -1),
            (2, -6, 8, -6, 2),
            (-2, 8, -12, 8, -2),
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
        ),
    ),
    (
        "edge5-h-rev",
        "Db",
        (
            (0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0),
            (-2, 8, -12, 8, -2),
            (2, -6, 8, -6, 2),
            (-1, 2, -2, 2, -1),
        ),
    ),
    (
        "edge5-v",
        "Dl",
        (
            (-1, 2, -2, 0, 0),
            (2, -6, 8, 0, 0),
            (-2, 8, -12, 0, 0),
            (2, -6, 8, 0, 0),
            (-1, 2, -2, 0, 0),
        ),
    ),
    (
        "edge5-v-rev",
        "Dr",
        (
            (0, 0, -2, 2, -1),
            (0, 0, 8, -6, 2),
            (0, 0, -12, 8, -2),
            (0, 0, 8, -6, 2),
            (0, 0, -2, 2, -1),
        ),
    ),
)

# Textbook kernels as cross-correlation taps (positive where intensity grows
# to the right for sobel-x, downwards for sobel-y).
_LAPLACIAN: _Grid = ((0, 1, 0), (1, -4, 1), (0, 1, 0))
_SOBEL_X: _Grid = ((-1, 0, 1), (-2, 0, 2), (-1, 0, 1))
_SOBEL_Y: _Grid = ((-1, -2, -1), (0, 0, 0), (1, 2, 1))

# The 30 SRM base residuals, in SRM.m's extraction order (all1st, all2nd,
# all3rd, all3x3, all5x5): 8 first-order, 4 second-order, 8 third-order,
# SQUARE 3x3, 4 EDGE 3x3, SQUARE 5x5 and 4 EDGE 5x5.
_SRM30: tuple[str, ...] = (
    *(name for name, _, _ in _FIRST_ORDER),
    *(name for name, _, _ in _SECOND_ORDER),
    *(name for name, _, _ in _THIRD_ORDER),
    "square3",
    *(name for name, _, _ in _EDGE3),
    "square5",
    *(name for name, _, _ in _EDGE5),
)

# The RGB-N kernels, in the order of the authors' code (c[0], c[1], c[2]).
_RGBN3 = ("square5", "square3", "second-order-h")


# --- construction ----------------------------------------------------------------


def _from_offsets(offsets: _Offsets) -> _Grid:
    grid = [[0] * _SIZE for _ in range(_SIZE)]
    for (di, dj), coefficient in offsets.items():
        grid[_CENTRE + di][_CENTRE + dj] = coefficient
    return tuple(tuple(row) for row in grid)


def _embed(grid: _Grid) -> _Grid:
    pad = (_SIZE - len(grid)) // 2
    zeros = (0,) * _SIZE
    rows = tuple((0,) * pad + row + (0,) * pad for row in grid)
    return (zeros,) * pad + rows + (zeros,) * pad


def _scaled(grid: _Grid, divisor: int) -> _Taps:
    return tuple(tuple(value / divisor for value in row) for row in grid)


def _build() -> dict[str, KernelInfo]:
    kernels: dict[str, KernelInfo] = {}

    def add(name: str, grid: _Grid, divisor: int, source: str, normalisation: str) -> None:
        kernels[name] = KernelInfo(name, _scaled(_embed(grid), divisor), source, normalisation)

    def srm(name: str, grid: _Grid, c: int, where: str) -> None:
        source = f"{_FK12}, {where}"
        if name in _RGBN3:
            source = f"{source}; {_ZHMD18}"
        add(name, grid, c, source, _srm_normalisation(c))

    for name, expression, offsets in _FIRST_ORDER:
        where = f"Sec. II-B and Fig. 2 (1a-1h); {_SRM_M} all1st: {expression}"
        srm(name, _from_offsets(offsets), 1, where)
    for name, expression, offsets in _SECOND_ORDER:
        where = f"Sec. II-B and Fig. 2 (2a-2e); {_SRM_M} Residual order 2 {expression}"
        srm(name, _from_offsets(offsets), 2, where)
    for name, expression, offsets in _THIRD_ORDER:
        where = f"Sec. II-B and Fig. 2 (3a-3h); {_SRM_M} all3rd: {expression}"
        srm(name, _from_offsets(offsets), 3, where)
    srm("square3", _S3A, 4, f"Sec. II-B and Fig. 2 (S3a); {_SRM_M} Residual 'KB'")
    for name, half, grid in _EDGE3:
        where = f"Sec. II-B and Fig. 2 (E3a-E3d); {_SRM_M} Residual 'edge-h'/'edge-v' {half}"
        srm(name, grid, 4, where)
    srm("square5", _S5A, 12, f"Sec. II-B and Fig. 2 (S5a); {_SRM_M} Residual 'KV'")
    for name, half, grid in _EDGE5:
        where = f"Sec. II-B and Fig. 2 caption (E5a-E5d); {_SRM_M} all5x5 {half}"
        srm(name, grid, 12, where)
    add("laplacian", _LAPLACIAN, 1, _GW08, _TEXTBOOK_NORMALISATION)
    add("sobel-x", _SOBEL_X, 1, f"{_SOBEL68}; d/dx (columns)", _TEXTBOOK_NORMALISATION)
    add("sobel-y", _SOBEL_Y, 1, f"{_SOBEL68}; d/dy (rows)", _TEXTBOOK_NORMALISATION)
    return kernels


KERNELS: Mapping[str, KernelInfo] = MappingProxyType(_build())
"""Every kernel by name. Read-only; the order is the ``srm30`` order, then
``laplacian``, ``sobel-x`` and ``sobel-y``."""

BANKS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "srm30": _SRM30,
        "rgbn3": _RGBN3,
        "square3": ("square3",),
        "square5": ("square5",),
        "laplacian": ("laplacian",),
        "sobel": ("sobel-x", "sobel-y"),
    }
)
"""Named kernel sets. Read-only; each value is a tuple of `KERNELS` names."""


# --- lookup --------------------------------------------------------------------


def _unknown(kind: str, name: str, known: Iterable[str]) -> str:
    candidates = sorted(known)
    matches = difflib.get_close_matches(name, candidates, n=3)
    if matches:
        quoted = [repr(match) for match in matches]
        options = quoted[0] if len(quoted) == 1 else ", ".join(quoted[:-1]) + " or " + quoted[-1]
        return f"unknown {kind} {name!r}; did you mean {options}?"
    return f"unknown {kind} {name!r}; known {kind}s: {', '.join(candidates)}"


def _resolve(bank_or_names: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(bank_or_names, str):
        if bank_or_names in BANKS:
            return BANKS[bank_or_names]
        if bank_or_names in KERNELS:
            raise KeyError(
                f"unknown bank {bank_or_names!r}; {bank_or_names!r} is a kernel name, "
                f"so pass it in a sequence: ({bank_or_names!r},)"
            )
        raise KeyError(_unknown("bank", bank_or_names, BANKS))
    names = tuple(bank_or_names)
    if not names:
        raise ValueError("srm_kernels needs at least one kernel name; got an empty sequence")
    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"kernel names must be strings; got {name!r}")
        if name not in KERNELS:
            raise KeyError(_unknown("kernel", name, KERNELS))
    return names


def srm_kernels(
    bank_or_names: str | Sequence[str],
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Stack a bank, or a list of named kernels, into a ``[K, 5, 5]`` tensor.

    Each call returns a new tensor, built from the exact float64 taps and
    rounded once to ``dtype``.

    Args:
        bank_or_names: A bank name from `BANKS` (``"srm30"``, ``"rgbn3"``,
            ``"square3"``, ``"square5"``, ``"laplacian"`` or ``"sobel"``), or
            a sequence of `KERNELS` names, stacked in the given order.
        dtype: A floating-point dtype; ``None`` means
            ``torch.get_default_dtype()``.
        device: The device to build on; ``None`` means the CPU.

    Returns:
        The kernels, shape ``[K, 5, 5]``, as cross-correlation taps (the
        convention of ``torch.nn.functional.conv2d``).

    Raises:
        KeyError: An unknown bank or kernel name. The message suggests the
            closest known names.
        ValueError: ``bank_or_names`` is an empty sequence.
        TypeError: ``dtype`` is not a floating-point dtype, or a kernel name
            is not a string.

    Example:
        >>> import torch
        >>> srm_kernels("srm30").shape
        torch.Size([30, 5, 5])
        >>> srm_kernels(("first-order-h",), dtype=torch.float64)[0, 2].tolist()
        [0.0, 0.0, -1.0, 1.0, 0.0]
        >>> srm_kernels("srm03")
        Traceback (most recent call last):
            ...
        KeyError: "unknown bank 'srm03'; did you mean 'srm30'?"
    """
    names = _resolve(bank_or_names)
    resolved = torch.get_default_dtype() if dtype is None else dtype
    if not resolved.is_floating_point:
        raise TypeError(f"srm_kernels needs a floating-point dtype; got {resolved}")
    taps = [KERNELS[name].taps for name in names]
    return torch.tensor(taps, dtype=torch.float64).to(dtype=resolved, device=device)
