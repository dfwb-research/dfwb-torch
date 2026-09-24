# dfwb-torch-srm

Fixed (optionally learnable, but always high-pass) **SRM / high-pass residual filters** for
PyTorch: the standard noise-residual front end in steganalysis and image forensics
(Fridrich & Kodovský 2012; RGB-N, Zhou et al. 2018; many deepfake detectors). These kernels
are usually copied between repos as ad-hoc `.npy` files with undocumented normalisation --
this package gives them named, cited, tested kernel banks instead.

Standalone PyTorch package: it installs, imports and runs with nothing but
`torch`. Part of [dfwb-torch](https://github.com/dfwb-research/dfwb-torch),
and optionally self-registers with
[Deepfake Workbench](https://github.com/dfwb-research/deepfake-workbench)
through the `dfwb.plugins` entry point when both are installed.

## Install

```bash
pip install dfwb-torch-srm
```

## Quickstart

```python
import torch
from dfwb_torch_srm import SRMConv2d

layer = SRMConv2d(in_channels=3, bank="srm30", mode="gray")
x = torch.rand(1, 3, 256, 256)
y = layer(x)
print(y.shape)  # torch.Size([1, 30, 256, 256])
```

See `examples/quickstart.py` for a runnable copy.

## Kernel banks (v0.1)

| Bank | Kernels | Source |
|---|---|---|
| `srm30` | The 30 SRM base residual kernels (8 first-order, 4 second-order, 8 third-order, SQUARE 3x3/5x5, 4+4 EDGE 3x3/5x5), 5x5 zero-embedded, each divided by its centre coefficient `c` | Fridrich & Kodovský, *Rich Models for Steganalysis of Digital Images*, IEEE TIFS 7(3):868-882, 2012, [doi:10.1109/TIFS.2012.2190402](https://doi.org/10.1109/TIFS.2012.2190402) |
| `rgbn3` | SQUARE 5x5 / 12, SQUARE 3x3 / 4, 2nd-order horizontal / 2, in the authors' code order | Zhou, Han, Morariu & Davis, *Learning Rich Features for Image Manipulation Detection*, CVPR 2018, pp. 1053-1061, [doi:10.1109/CVPR.2018.00116](https://doi.org/10.1109/CVPR.2018.00116); [RGB-N code](https://github.com/pengzhou1108/RGB-N)@`fdc612d` |
| `square3`, `square5` | The single SQUARE 3x3 (÷4) / SQUARE 5x5 (÷12) kernels | Fridrich & Kodovský 2012 |
| `laplacian`, `sobel` | Textbook 4-neighbour Laplacian / 3x3 Sobel gradient (non-SRM, for ablations) | Gonzalez & Woods, *Digital Image Processing*, 3rd ed., 2008, Sec. 3.6.2; Sobel & Feldman 1968 |

Every kernel is zero-sum (DC gain 0). The exact source section, URL and commit used for each
bank's normalisation, and every place the sources disagree, is recorded in
[`docs/kernels.md`](docs/kernels.md); `tests/test_reference.py` pins every kernel to a
transcribed reference table, so any change is a deliberate, reviewed diff.

## Modes

```
mode="depthwise" (default)        C*K out
  x[B,C,H,W] --every kernel, every channel-->  y[B, C*K, H', W']
  channel-major: c0k0, c0k1, ..., c0k{K-1}, c1k0, ...

mode="sum"                         K out
  x[B,C,H,W] --every kernel, summed over C-->  y[B, K, H', W']

mode="gray"                        K out
  x[B,3,H,W] --luminance (0.299R+0.587G+0.114B)-->  [B,1,H,W] --every kernel-->  y[B, K, H', W']

mode="per-channel"                 C out   (needs K == C)
  x[B,C,H,W] --kernel i on channel i-->  y[B, C, H', W']
```

`H'`, `W'` follow `padding`: unchanged for `padding="same"` (the default), or `H-4, W-4` for
`padding="valid"` (kernels are always 5x5).

## Pitfalls

- **Input range.** Truncation (`truncate`, the TLU threshold) is expressed in 0-255 residual
  units, as in the SRM literature. If your images are scaled to `[0, 1]` (the default,
  `input_scale="0-1"`), the effective threshold is `truncate / 255`, so the clamp means the
  same thing either way. Pass `input_scale="0-255"` if you feed 0-255 images directly; mixing
  the two up silently over- or under-truncates.
- **`mode="per-channel"` needs `K == C`.** It applies kernel `i` to channel `i`, so the bank's
  kernel count `K` must equal `in_channels`. A mismatch raises a `ValueError` naming both `K`
  and `C` at construction time, not later at the first forward pass.

## Using with Deepfake Workbench

`dfwb_torch_srm` optionally self-registers as the framework's `layers/srm` component through
the `dfwb.plugins` entry point (`register(api)` in `dfwb_plugin.py`), with no dependency the
other way: this package never imports `dfwb`, and works standalone even if it's never
installed. Deepfake Workbench's model stem hook (arriving in its own M4 milestone) is intended
to build the registered layer from a config block, for example:

```yaml
model:
  stem:
    name: srm
    bank: srm30
    mode: gray
```

## Citation

If you use these kernels, please cite the original papers:

```bibtex
@article{fridrich2012rich,
  author  = {Fridrich, Jessica and Kodovsk{\'y}, Jan},
  title   = {Rich Models for Steganalysis of Digital Images},
  journal = {IEEE Transactions on Information Forensics and Security},
  volume  = {7},
  number  = {3},
  pages   = {868--882},
  year    = {2012},
  doi     = {10.1109/TIFS.2012.2190402},
}

@inproceedings{zhou2018learning,
  author    = {Zhou, Peng and Han, Xintong and Morariu, Vlad I. and Davis, Larry S.},
  title     = {Learning Rich Features for Image Manipulation Detection},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {1053--1061},
  year      = {2018},
  doi       = {10.1109/CVPR.2018.00116},
}
```

If `dfwb-torch-srm` itself helps your work, please consider citing it too -- see
[`CITATION.cff`](CITATION.cff).

## Licence

MIT. See `LICENSE`.
