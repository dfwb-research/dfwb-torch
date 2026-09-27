# Kernel reference

Every kernel in `dfwb_torch_srm.kernels`: its name, orientation, source and
normalisation, and where the sources disagree. The tests pin each kernel to a
hand-transcribed table in `tests/_tables.py`, so any change to a kernel is a
deliberate, reviewed diff.

## Conventions

- **Taps are cross-correlation taps**, the convention of
  `torch.nn.functional.conv2d`. For the output at pixel `(i, j)`, tap
  `taps[r][c]` multiplies `X[i + r − 2, j + c − 2]`, so row 2, column 2
  multiplies the centre pixel. Row indices grow downwards and column indices
  grow to the right.
- **Every kernel is 5×5.** 3×3 kernels are zero-embedded in the centre.
- **Every kernel is zero-sum** (DC gain 0).
- **Orientation suffixes:**
  - `-h` runs along the row, and `-v` along the column (the transpose of `-h`).
  - `-d1` runs along the main diagonal, towards the bottom right, and `-d2`
    along the anti-diagonal, towards the bottom left (the left-right mirror
    of `-d1`).
  - `-rev` is the same kernel rotated by 180°. For a first- or third-order
    residual that is the opposite direction. For an EDGE kernel it is the
    opposite half.
  - For an EDGE kernel, `-h` is the half above the centre row, as E3a is drawn
    in Fridrich & Kodovský (2012), Fig. 2, and `-v` is the left half.

## Normalisation

SRM writes every residual as `R = X̂(N) − c·X`, where `c` is the centre
coefficient (Fridrich & Kodovský 2012, Sec. II-A, Eq. (1)). Each `srm30`
kernel is **divided by `c`** (1, 2, 3, 4 or 12). Its centre tap is then −1,
and residual magnitudes are comparable across orders.

This is SRM's own normalisation at its finest quantisation step:

- Eq. (2) quantises `R / q`.
- Sec. II-D, Eq. (9) takes `q ∈ {c, 1.5c, 2c}` (and `{1, 2}` for `c = 1`).
- The authors' `SRM.m` (lines 39–44) calls each extractor with `q·c` for
  `q = 1, 1.5, 2`. Its comment on `all1st`/`all3rd` reads: *"Recommended
  values of q are c, 1.5c, 2c, where c is the central coefficient in the
  differential (at X(I,J))."*

This package uses `q = c`. The coarser steps are a quantiser setting, not a
different kernel.

The `rgbn3` kernels are RGB-N's exactly: SQUARE 5×5 ÷ 12, SQUARE 3×3 ÷ 4 and
second-order horizontal ÷ 2. By the rule above, these are the same `srm30`
kernels, and `test_rgbn3_matches_zhou_2018` checks that they match to the bit.
`laplacian`, `sobel-x` and `sobel-y` keep their textbook integer coefficients.

## Sources

| Key | Source | What was used |
|---|---|---|
| FK12 | J. Fridrich and J. Kodovský, "Rich Models for Steganalysis of Digital Images", *IEEE TIFS* 7(3):868–882, 2012, [doi:10.1109/TIFS.2012.2190402](https://doi.org/10.1109/TIFS.2012.2190402) ([author PDF](http://dde.binghamton.edu/kodovsky/pdf/TIFS2012-SRM.pdf)) | Sec. II-A, Eq. (1) and (2); Sec. II-B and Fig. 2 (1a–1h, 2a–2e, S3a, E3a–E3d, S5a; 3a–3h and E5a–E5d are defined in the text and caption); Sec. II-D, Eq. (9) |
| SRM.m | The authors' MATLAB extractor, `SRM.m` in [SRM.zip](http://dde.binghamton.edu/download/feature_extractors/download/SRM.zip), from [dde.binghamton.edu/download/feature_extractors](http://dde.binghamton.edu/download/feature_extractors/). The site has no version control, so this is `SRM.m` with sha256 `b11272825ffb9c296505cda1888dc6702c21c9163b93cf8c3fe67c3854c1aa3e` (header dated October 2011; zip entry 2012-03-06) | `all1st` and `all3rd` (the eight directions R, L, U, D, RU, LU, RD, LD); `Residual` (order 2 `'hor'`, `'ver'`, `'diag'`, `'mdiag'`; `'KB'`; `'edge-h'` Du/Db; `'edge-v'` Dl/Dr; `'KV'`); `all5x5` (EDGE 5×5 Du, Dr, Db, Dl); lines 39–44 (the quantisation steps) |
| ZHMD18 | P. Zhou, X. Han, V. I. Morariu and L. S. Davis, "Learning Rich Features for Image Manipulation Detection", *CVPR* 2018, pp. 1053–1061, [doi:10.1109/CVPR.2018.00116](https://doi.org/10.1109/CVPR.2018.00116) ([CVF PDF](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhou_Learning_Rich_Features_CVPR_2018_paper.pdf)) | Sec. 3.2 ("Noise Stream") and Fig. 4 |
| RGB-N code | [github.com/pengzhou1108/RGB-N](https://github.com/pengzhou1108/RGB-N) at commit [`fdc612d781a83c1783e46ab1e76bb1cc6aa0f458`](https://github.com/pengzhou1108/RGB-N/blob/fdc612d781a83c1783e46ab1e76bb1cc6aa0f458/lib/nets/resnet_fusion.py#L176-L194) | `lib/nets/resnet_fusion.py`, lines 176–194 (`c[0] … /12`, `c[1] … /4`, `c[2] … /2`), the network the README trains (`res101_fusion`) |
| YNY17 | J. Ye, J. Ni and Y. Yi, "Deep Learning Hierarchical Representations for Image Steganalysis", *IEEE TIFS* 12(11):2545–2557, 2017, [doi:10.1109/TIFS.2017.2710946](https://doi.org/10.1109/TIFS.2017.2710946) | The 30-kernel SRM set (8 + 4 + 8 + 1 + 4 + 1 + 4) used as a CNN front end. Checked against the public reimplementation [github.com/Caenorst/YeNet-Pytorch](https://github.com/Caenorst/YeNet-Pytorch) at commit [`02703fc43360a09487a1232622c98bc6a545c9db`](https://github.com/Caenorst/YeNet-Pytorch/blob/02703fc43360a09487a1232622c98bc6a545c9db/SRM_Kernels.npy), `SRM_Kernels.npy`. That repository is not the authors' own; the paper's own text was not available to check |
| GW08 | R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 3rd ed., Pearson, 2008, Sec. 3.6.2 | The 4-neighbour Laplacian `f(x+1,y) + f(x−1,y) + f(x,y+1) + f(x,y−1) − 4f(x,y)` |
| Sobel68 | I. Sobel and G. Feldman, "A 3×3 Isotropic Gradient Operator for Image Processing", talk at the Stanford Artificial Intelligence Project, 1968 (history: I. Sobel, "History and Definition of the Sobel Operator", 2014) | The 3×3 Sobel gradient kernels |

## `srm30`: the 30 SRM base residuals

The bank is in `SRM.m`'s extraction order: 8 first-order, 4 second-order,
8 third-order, SQUARE 3×3, 4 EDGE 3×3, SQUARE 5×5 and 4 EDGE 5×5. The
**divisor** is `c`. The last column gives each kernel's index in the YeNet
reimplementation's `SRM_Kernels.npy`: that file holds the same 30 kernels as
unnormalised integers, so its kernel at that index, divided by `c`, equals
ours.

| # | Name | Orientation | `SRM.m` residual | Divisor `c` | FK12 Fig. 2 | YeNet npy index |
|---|---|---|---|---|---|---|
| 0 | `first-order-h` | → (right) | `all1st` `R = X(I,J+1)-X(I,J)` | 1 | 1a–1h | 2 |
| 1 | `first-order-h-rev` | ← (left) | `all1st` `L = X(I,J-1)-X(I,J)` | 1 | 1a–1h | 6 |
| 2 | `first-order-v` | ↓ (down) | `all1st` `D = X(I+1,J)-X(I,J)` | 1 | 1a–1h | 4 |
| 3 | `first-order-v-rev` | ↑ (up) | `all1st` `U = X(I-1,J)-X(I,J)` | 1 | 1a–1h | 0 |
| 4 | `first-order-d1` | ↘ (down-right) | `all1st` `RD = X(I+1,J+1)-X(I,J)` | 1 | 1a–1h | 3 |
| 5 | `first-order-d1-rev` | ↖ (up-left) | `all1st` `LU = X(I-1,J-1)-X(I,J)` | 1 | 1a–1h | 7 |
| 6 | `first-order-d2` | ↙ (down-left) | `all1st` `LD = X(I+1,J-1)-X(I,J)` | 1 | 1a–1h | 5 |
| 7 | `first-order-d2-rev` | ↗ (up-right) | `all1st` `RU = X(I-1,J+1)-X(I,J)` | 1 | 1a–1h | 1 |
| 8 | `second-order-h` | row (─) | `Residual(X,2,'hor')` `X(I,J-1)-2*X(I,J)+X(I,J+1)` | 2 | 2a–2e | 10 |
| 9 | `second-order-v` | column (│) | `Residual(X,2,'ver')` `X(I-1,J)-2*X(I,J)+X(I+1,J)` | 2 | 2a–2e | 8 |
| 10 | `second-order-d1` | main diagonal (╲) | `Residual(X,2,'diag')` `X(I-1,J-1)-2*X(I,J)+X(I+1,J+1)` | 2 | 2a–2e | 11 |
| 11 | `second-order-d2` | anti-diagonal (╱) | `Residual(X,2,'mdiag')` `X(I-1,J+1)-2*X(I,J)+X(I+1,J-1)` | 2 | 2a–2e | 9 |
| 12 | `third-order-h` | → (right) | `all3rd` `R = -X(I,J+2)+3*X(I,J+1)-3*X(I,J)+X(I,J-1)` | 3 | 3a–3h (caption) | 14 |
| 13 | `third-order-h-rev` | ← (left) | `all3rd` `L = -X(I,J-2)+3*X(I,J-1)-3*X(I,J)+X(I,J+1)` | 3 | 3a–3h (caption) | 18 |
| 14 | `third-order-v` | ↓ (down) | `all3rd` `D = -X(I+2,J)+3*X(I+1,J)-3*X(I,J)+X(I-1,J)` | 3 | 3a–3h (caption) | 16 |
| 15 | `third-order-v-rev` | ↑ (up) | `all3rd` `U = -X(I-2,J)+3*X(I-1,J)-3*X(I,J)+X(I+1,J)` | 3 | 3a–3h (caption) | 12 |
| 16 | `third-order-d1` | ↘ (down-right) | `all3rd` `RD = -X(I+2,J+2)+3*X(I+1,J+1)-3*X(I,J)+X(I-1,J-1)` | 3 | 3a–3h (caption) | 15 |
| 17 | `third-order-d1-rev` | ↖ (up-left) | `all3rd` `LU = -X(I-2,J-2)+3*X(I-1,J-1)-3*X(I,J)+X(I+1,J+1)` | 3 | 3a–3h (caption) | 19 |
| 18 | `third-order-d2` | ↙ (down-left) | `all3rd` `LD = -X(I+2,J-2)+3*X(I+1,J-1)-3*X(I,J)+X(I-1,J+1)` | 3 | 3a–3h (caption) | 17 |
| 19 | `third-order-d2-rev` | ↗ (up-right) | `all3rd` `RU = -X(I-2,J+2)+3*X(I-1,J+1)-3*X(I,J)+X(I+1,J-1)` | 3 | 3a–3h (caption) | 13 |
| 20 | `square3` | all around (non-directional) | `Residual(X,2,'KB')` | 4 | S3a | 20 |
| 21 | `edge3-h` | upper half (rows −1…0) | `Residual(X,2,'edge-h')` `Du` | 4 | E3a–E3d | 21 |
| 22 | `edge3-h-rev` | lower half (rows 0…+1) | `Residual(X,2,'edge-h')` `Db` | 4 | E3a–E3d | 23 |
| 23 | `edge3-v` | left half (columns −1…0) | `Residual(X,2,'edge-v')` `Dl` | 4 | E3a–E3d | 24 |
| 24 | `edge3-v-rev` | right half (columns 0…+1) | `Residual(X,2,'edge-v')` `Dr` | 4 | E3a–E3d | 22 |
| 25 | `square5` | all around (non-directional) | `Residual(X,3,'KV')` | 12 | S5a | 25 |
| 26 | `edge5-h` | upper half (rows −2…0) | `all5x5` `Du` | 12 | E5a–E5d (caption) | 26 |
| 27 | `edge5-h-rev` | lower half (rows 0…+2) | `all5x5` `Db` | 12 | E5a–E5d (caption) | 28 |
| 28 | `edge5-v` | left half (columns −2…0) | `all5x5` `Dl` | 12 | E5a–E5d (caption) | 29 |
| 29 | `edge5-v-rev` | right half (columns 0…+2) | `all5x5` `Dr` | 12 | E5a–E5d (caption) | 27 |

## `rgbn3`: RGB-N's noise-stream kernels

The kernels are in the order of the authors' code (channel `i` of their
`Wcnn[:, :, :, i]` is `c[i]`). This is the order a checkpoint trained with that
code expects.

| # | Name | RGB-N code (`resnet_fusion.py`, fdc612d) | Divisor | Paper Fig. 4 position |
|---|---|---|---|---|
| 0 | `square5` | `c[0]` (lines 177–178) | 12 | 2nd |
| 1 | `square3` | `c[1]` (lines 180–189) | 4 | 1st |
| 2 | `second-order-h` | `c[2]` (lines 191–194) | 2 | 3rd |

In RGB-N, every one of the three output channels applies its kernel to all
three input channels and sums the results (`Wcnn[:, :, 0..2, i] = c[i]`),
which is `srm_conv2d(..., mode="sum")` in this package.

## Single-kernel and textbook banks

| Bank | Kernels | Source | Normalisation |
|---|---|---|---|
| `square3` | `square3` | FK12 (S3a); `SRM.m` `'KB'` | ÷ 4 (as in `srm30`) |
| `square5` | `square5` | FK12 (S5a); `SRM.m` `'KV'` | ÷ 12 (as in `srm30`) |
| `laplacian` | `laplacian`: `[[0, 1, 0], [1, −4, 1], [0, 1, 0]]` | GW08 | none (integer coefficients) |
| `sobel` | `sobel-x`: `[[−1, 0, 1], [−2, 0, 2], [−1, 0, 1]]`; `sobel-y`: `[[−1, −2, −1], [0, 0, 0], [1, 2, 1]]` | Sobel68 | none (integer coefficients) |

`sobel-x` is positive where intensity increases to the right, and `sobel-y`
where it increases downwards. Both are ablation baselines, not SRM residuals.

## Where the sources disagree

1. **Order of the RGB-N kernels.** The paper's Fig. 4 prints them as SQUARE
   3×3 ÷ 4, SQUARE 5×5 ÷ 12 and second-order ÷ 2. The authors' code uses
   SQUARE 5×5, SQUARE 3×3, second-order (`c[0]`, `c[1]`, `c[2]`). The values
   are the same; only the channel order differs. **Chosen:** the code order,
   because it is the order a trained RGB-N expects.
2. **Orientation of RGB-N's second-order kernel.** `resnet_fusion.py` and
   `resnet_fusion_noise.py` (lines 198–201) use the horizontal kernel
   (`c[2][2][1..3] = 1, −2, 1`), and so does the paper's Fig. 4. The
   noise-stream-only ablation `resnet_v1_noise.py` (lines 102–105) uses the
   vertical one (`c[2][1..3][2]`). **Chosen:** horizontal, as in the paper
   and the network the README trains (`res101_fusion`).
3. **Direction of SRM.m's first-order `'mdiag'`.** `Residual(X, 1, 'mdiag')`
   is `X(I−1,J+1) − X(I,J)`, pointing up-right, but `Residual(X, 3, 'mdiag')`
   points down-left. This package names kernels by geometry: `-d2` points
   down-left for every order. So SRM.m's first-order `'mdiag'` is
   `first-order-d2-rev` here. The extractor itself (`all1st`) spells out all
   eight directions and never calls `Residual(X, 1, …)`, so the content of
   `srm30` is unaffected.
4. **"Normalised by its quantisation step" versus "divided by the centre
   coefficient".** Both phrases describe this package's kernels: they mean
   the same thing, because SRM's finest quantisation step is `q = c` (see
   [Normalisation](#normalisation)).
5. **Scaling in public CNN code.** YeNet-Pytorch's `SRM_Kernels.npy` stores
   the 30 kernels as unnormalised integers (centre −1, −2, −3, −4 or −12), in
   a different order. `srm30` equals those kernels divided by `c`; the table
   above maps the indices. If you load weights trained with the integer
   kernels, multiply by `c`, or retrain.
6. **EDGE 5×5 is not drawn in the paper.** Fig. 2's caption defines E5a–E5d
   as the corresponding halves of S5a, analogous to E3a–E3d. They were
   transcribed from `SRM.m` `all5x5` (Du, Db, Dl, Dr), and each is the
   stated half of S5a.
7. **Excluded:** `SRM.m`'s `Residual` also defines `'edge-d'` and `'edge-m'`
   (diagonal EDGE 3×3 halves with centre coefficient −1). Neither the
   extractor (`all3x3`) nor the paper uses them, so they are not SRM base
   residuals, and they are not included. SRM's non-linear min/max residuals,
   co-occurrences and truncation (`T = 2`) are not kernels, and are out of
   scope for this module. Truncation lives in the functional API.
8. **Sign conventions of the textbook kernels.** Some texts write the
   Laplacian with a centre of +4, and some write Sobel as a convolution
   kernel, which is flipped relative to these correlation taps. This package
   uses the centre −4 Laplacian of GW08, and Sobel correlation taps that are
   positive along +x and +y. Both match `scipy.ndimage.laplace` and
   `scipy.ndimage.sobel` (checked by impulse response against SciPy 1.18.1).
