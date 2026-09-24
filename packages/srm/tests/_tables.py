"""Transcribed reference tables for every kernel in dfwb_torch_srm.kernels.

Each entry is ``name -> (divisor, grid)``: ``grid`` is the integer 5x5 kernel
exactly as printed in the source (3x3 kernels zero-embedded in the centre),
and the expected taps are ``grid[r][c] / divisor``. Taps are applied as a
cross-correlation, like ``torch.nn.functional.conv2d``: the output at pixel
``(i, j)`` is ``sum(taps[r][c] * X[i + r - 2, j + c - 2])``, so row 2,
column 2 multiplies the centre pixel ``X[i, j]``.

These tables are the test oracle. They were transcribed by hand from the
sources below, independently of ``kernels.py``, so a change to a kernel is a
deliberate diff in both places.

Sources:

* [FK12] J. Fridrich and J. Kodovsky, "Rich Models for Steganalysis of
  Digital Images", IEEE TIFS 7(3):868-882, 2012, doi:10.1109/TIFS.2012.2190402.
  Sec. II-A Eq. (1) (``R_ij = Xhat_ij - c X_ij``, c = the centre
  coefficient), Sec. II-B and Fig. 2 (1a, 2a, S3a, E3a, S5a; the 3rd order
  "replacing X_{i,j+1} - X_ij with -X_{i,j+2} + 3X_{i,j+1} - 3X_ij + X_{i,j-1}";
  E5a-E5d built from S5a "in an analogical manner as E3a-E3d").
* [SRM.m] The authors' MATLAB extractor, SRM.m in
  http://dde.binghamton.edu/download/feature_extractors/download/SRM.zip
  (SRM.m sha256 b11272825ffb9c296505cda1888dc6702c21c9163b93cf8c3fe67c3854c1aa3e):
  ``all1st``/``all3rd`` (the eight directions R, L, U, D, RU, LU, RD, LD),
  ``Residual`` (order 2: 'hor', 'ver', 'diag', 'mdiag'; 'KB'; 'edge-h' Du/Db;
  'edge-v' Dl/Dr; 'KV') and ``all5x5`` (EDGE5x5 Du, Dr, Db, Dl).
* [ZHMD18] P. Zhou, X. Han, V. I. Morariu and L. S. Davis, "Learning Rich
  Features for Image Manipulation Detection", CVPR 2018, pp. 1053-1061,
  doi:10.1109/CVPR.2018.00116, Sec. 3.2 and Fig. 4; and the authors' code
  https://github.com/pengzhou1108/RGB-N at commit
  fdc612d781a83c1783e46ab1e76bb1cc6aa0f458, lib/nets/resnet_fusion.py
  lines 176-194 (``c[0] ... /12``, ``c[1] ... /4``, ``c[2] ... /2``).
* [Sobel68] I. Sobel and G. Feldman, "A 3x3 Isotropic Gradient Operator for
  Image Processing", talk at the Stanford Artificial Intelligence Project,
  1968 (see I. Sobel, "History and Definition of the Sobel Operator", 2014).
* [GW08] R. C. Gonzalez and R. E. Woods, Digital Image Processing, 3rd ed.,
  Pearson, 2008, Sec. 3.6.2 (the 4-neighbour Laplacian).
"""

Grid = tuple[
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
    tuple[int, int, int, int, int],
]
Table = tuple[int, Grid]

# fmt: off
# The SQUARE 5x5 kernel S5a, [FK12] Fig. 2 and [SRM.m] Residual 'KV'. Named so
# the EDGE5x5 halves below can be checked against it by eye.
_S5A: Grid = (
    ( -1,   2,  -2,   2,  -1),
    (  2,  -6,   8,  -6,   2),
    ( -2,   8, -12,   8,  -2),
    (  2,  -6,   8,  -6,   2),
    ( -1,   2,  -2,   2,  -1),
)

# The 30 SRM base residuals, in the order of BANKS["srm30"]. The divisor is
# the magnitude of the centre coefficient c ([FK12] Eq. (1)), i.e. the
# smallest SRM quantisation step q = c ([FK12] Eq. (9); [SRM.m] lines 39-44).
SRM30_TABLES: dict[str, Table] = {
    # --- 1st order, c = 1. [FK12] Fig. 2 (1a); [SRM.m] all1st. ---
    # R = X(I,J+1) - X(I,J)
    "first-order-h": (1, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0, -1,  1,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # L = X(I,J-1) - X(I,J)
    "first-order-h-rev": (1, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  1, -1,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # D = X(I+1,J) - X(I,J)
    "first-order-v": (1, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0, -1,  0,  0),
        ( 0,  0,  1,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # U = X(I-1,J) - X(I,J)
    "first-order-v-rev": (1, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  1,  0,  0),
        ( 0,  0, -1,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # RD = X(I+1,J+1) - X(I,J)
    "first-order-d1": (1, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0, -1,  0,  0),
        ( 0,  0,  0,  1,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # LU = X(I-1,J-1) - X(I,J)
    "first-order-d1-rev": (1, (
        ( 0,  0,  0,  0,  0),
        ( 0,  1,  0,  0,  0),
        ( 0,  0, -1,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # LD = X(I+1,J-1) - X(I,J)
    "first-order-d2": (1, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0, -1,  0,  0),
        ( 0,  1,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # RU = X(I-1,J+1) - X(I,J)
    "first-order-d2-rev": (1, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  1,  0),
        ( 0,  0, -1,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # --- 2nd order, c = 2. [FK12] Fig. 2 (2a); [SRM.m] Residual order 2. ---
    # 'hor': X(I,J-1) - 2*X(I,J) + X(I,J+1)
    "second-order-h": (2, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  1, -2,  1,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # 'ver': X(I-1,J) - 2*X(I,J) + X(I+1,J)
    "second-order-v": (2, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  1,  0,  0),
        ( 0,  0, -2,  0,  0),
        ( 0,  0,  1,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # 'diag': X(I-1,J-1) - 2*X(I,J) + X(I+1,J+1)
    "second-order-d1": (2, (
        ( 0,  0,  0,  0,  0),
        ( 0,  1,  0,  0,  0),
        ( 0,  0, -2,  0,  0),
        ( 0,  0,  0,  1,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # 'mdiag': X(I-1,J+1) - 2*X(I,J) + X(I+1,J-1)
    "second-order-d2": (2, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  1,  0),
        ( 0,  0, -2,  0,  0),
        ( 0,  1,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # --- 3rd order, c = 3. [FK12] Sec. II-B; [SRM.m] all3rd. ---
    # R = -X(I,J+2) + 3*X(I,J+1) - 3*X(I,J) + X(I,J-1)
    "third-order-h": (3, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  1, -3,  3, -1),
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # L = -X(I,J-2) + 3*X(I,J-1) - 3*X(I,J) + X(I,J+1)
    "third-order-h-rev": (3, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
        (-1,  3, -3,  1,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # D = -X(I+2,J) + 3*X(I+1,J) - 3*X(I,J) + X(I-1,J)
    "third-order-v": (3, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  1,  0,  0),
        ( 0,  0, -3,  0,  0),
        ( 0,  0,  3,  0,  0),
        ( 0,  0, -1,  0,  0),
    )),
    # U = -X(I-2,J) + 3*X(I-1,J) - 3*X(I,J) + X(I+1,J)
    "third-order-v-rev": (3, (
        ( 0,  0, -1,  0,  0),
        ( 0,  0,  3,  0,  0),
        ( 0,  0, -3,  0,  0),
        ( 0,  0,  1,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # RD = -X(I+2,J+2) + 3*X(I+1,J+1) - 3*X(I,J) + X(I-1,J-1)
    "third-order-d1": (3, (
        ( 0,  0,  0,  0,  0),
        ( 0,  1,  0,  0,  0),
        ( 0,  0, -3,  0,  0),
        ( 0,  0,  0,  3,  0),
        ( 0,  0,  0,  0, -1),
    )),
    # LU = -X(I-2,J-2) + 3*X(I-1,J-1) - 3*X(I,J) + X(I+1,J+1)
    "third-order-d1-rev": (3, (
        (-1,  0,  0,  0,  0),
        ( 0,  3,  0,  0,  0),
        ( 0,  0, -3,  0,  0),
        ( 0,  0,  0,  1,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # LD = -X(I+2,J-2) + 3*X(I+1,J-1) - 3*X(I,J) + X(I-1,J+1)
    "third-order-d2": (3, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  1,  0),
        ( 0,  0, -3,  0,  0),
        ( 0,  3,  0,  0,  0),
        (-1,  0,  0,  0,  0),
    )),
    # RU = -X(I-2,J+2) + 3*X(I-1,J+1) - 3*X(I,J) + X(I+1,J-1)
    "third-order-d2-rev": (3, (
        ( 0,  0,  0,  0, -1),
        ( 0,  0,  0,  3,  0),
        ( 0,  0, -3,  0,  0),
        ( 0,  1,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # --- SQUARE 3x3 (S3a), c = 4. [FK12] Fig. 2 (S3a); [SRM.m] 'KB'. ---
    "square3": (4, (
        ( 0,  0,  0,  0,  0),
        ( 0, -1,  2, -1,  0),
        ( 0,  2, -4,  2,  0),
        ( 0, -1,  2, -1,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # --- EDGE 3x3 (E3a in four rotations), c = 4. [FK12] Fig. 2 (E3a);
    # [SRM.m] Residual 'edge-h' (Du, Db) and 'edge-v' (Dl, Dr). ---
    # Du, the upper half, as drawn in Fig. 2 (E3a)
    "edge3-h": (4, (
        ( 0,  0,  0,  0,  0),
        ( 0, -1,  2, -1,  0),
        ( 0,  2, -4,  2,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # Db, the lower half
    "edge3-h-rev": (4, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  2, -4,  2,  0),
        ( 0, -1,  2, -1,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # Dl, the left half
    "edge3-v": (4, (
        ( 0,  0,  0,  0,  0),
        ( 0, -1,  2,  0,  0),
        ( 0,  2, -4,  0,  0),
        ( 0, -1,  2,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # Dr, the right half
    "edge3-v-rev": (4, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  2, -1,  0),
        ( 0,  0, -4,  2,  0),
        ( 0,  0,  2, -1,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # --- SQUARE 5x5 (S5a), c = 12. [FK12] Fig. 2 (S5a); [SRM.m] 'KV'. ---
    "square5": (12, _S5A),
    # --- EDGE 5x5 (E5a in four rotations), c = 12. [FK12] Fig. 2 caption;
    # [SRM.m] all5x5 Du, Db, Dl, Dr. ---
    # Du, rows 0-2 of S5a
    "edge5-h": (12, (
        ( -1,   2,  -2,   2,  -1),
        (  2,  -6,   8,  -6,   2),
        ( -2,   8, -12,   8,  -2),
        (  0,   0,   0,   0,   0),
        (  0,   0,   0,   0,   0),
    )),
    # Db, rows 2-4 of S5a
    "edge5-h-rev": (12, (
        (  0,   0,   0,   0,   0),
        (  0,   0,   0,   0,   0),
        ( -2,   8, -12,   8,  -2),
        (  2,  -6,   8,  -6,   2),
        ( -1,   2,  -2,   2,  -1),
    )),
    # Dl, columns 0-2 of S5a
    "edge5-v": (12, (
        ( -1,   2,  -2,   0,   0),
        (  2,  -6,   8,   0,   0),
        ( -2,   8, -12,   0,   0),
        (  2,  -6,   8,   0,   0),
        ( -1,   2,  -2,   0,   0),
    )),
    # Dr, columns 2-4 of S5a
    "edge5-v-rev": (12, (
        (  0,   0,  -2,   2,  -1),
        (  0,   0,   8,  -6,   2),
        (  0,   0, -12,   8,  -2),
        (  0,   0,   8,  -6,   2),
        (  0,   0,  -2,   2,  -1),
    )),
}

# RGB-N's three kernels in the authors' code order: [ZHMD18] code
# lib/nets/resnet_fusion.py lines 176-194 at fdc612d (c[0], c[1], c[2]).
# The paper's Fig. 4 prints the same three kernels in the order c[1], c[0], c[2].
RGBN3_TABLES: dict[str, Table] = {
    # c[0] = [[-1,2,-2,2,-1],[2,-6,8,-6,2],[-2,8,-12,8,-2],[2,-6,8,-6,2],[-1,2,-2,2,-1]]; c[0]/12
    "c[0]": (12, (
        ( -1,   2,  -2,   2,  -1),
        (  2,  -6,   8,  -6,   2),
        ( -2,   8, -12,   8,  -2),
        (  2,  -6,   8,  -6,   2),
        ( -1,   2,  -2,   2,  -1),
    )),
    # c[1][1][1..3] = -1, 2, -1; c[1][2][1..3] = 2, -4, 2; c[1][3][1..3] = -1, 2, -1; c[1]/4
    "c[1]": (4, (
        ( 0,  0,  0,  0,  0),
        ( 0, -1,  2, -1,  0),
        ( 0,  2, -4,  2,  0),
        ( 0, -1,  2, -1,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # c[2][2][1] = 1; c[2][2][2] = -2; c[2][2][3] = 1; c[2]/2
    "c[2]": (2, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  1, -2,  1,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
}

# Textbook (non-SRM) kernels, unnormalised integer coefficients, written as
# cross-correlation taps.
TEXTBOOK_TABLES: dict[str, Table] = {
    # [GW08]: f(x+1,y) + f(x-1,y) + f(x,y+1) + f(x,y-1) - 4 f(x,y)
    "laplacian": (1, (
        ( 0,  0,  0,  0,  0),
        ( 0,  0,  1,  0,  0),
        ( 0,  1, -4,  1,  0),
        ( 0,  0,  1,  0,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # [Sobel68]: positive where intensity increases to the right (+x = +column)
    "sobel-x": (1, (
        ( 0,  0,  0,  0,  0),
        ( 0, -1,  0,  1,  0),
        ( 0, -2,  0,  2,  0),
        ( 0, -1,  0,  1,  0),
        ( 0,  0,  0,  0,  0),
    )),
    # [Sobel68]: positive where intensity increases downwards (+y = +row)
    "sobel-y": (1, (
        ( 0,  0,  0,  0,  0),
        ( 0, -1, -2, -1,  0),
        ( 0,  0,  0,  0,  0),
        ( 0,  1,  2,  1,  0),
        ( 0,  0,  0,  0,  0),
    )),
}
# fmt: on

ALL_TABLES: dict[str, Table] = {**SRM30_TABLES, **TEXTBOOK_TABLES}


def taps_of(table: Table) -> tuple[tuple[float, ...], ...]:
    """Return a table's expected taps: every integer divided by the divisor."""
    divisor, grid = table
    return tuple(tuple(value / divisor for value in row) for row in grid)
