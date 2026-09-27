# Changelog

All notable changes to `dfwb-torch-srm` are documented here.

## [0.1.0rc1]

- `srm_kernels`, `BANKS` and `KERNELS`: the 30 canonical SRM base residual kernels (`srm30`),
  the three RGB-N noise-stream kernels (`rgbn3`), single-kernel banks (`square3`, `square5`)
  and textbook ablation kernels (`laplacian`, `sobel`), each cited and normalised against its
  source (see `docs/kernels.md`).
- `srm_conv2d` / `SRMConv2d`: batched SRM/high-pass convolution with `depthwise`, `sum`,
  `gray` and `per-channel` modes, a TLU truncation stage with an `input_scale`-aware
  threshold, and an optional `trainable=True` mode that keeps every kernel zero-DC via a
  `torch.nn.utils.parametrize` parametrisation.
- `TLU`: a standalone truncated-linear-unit module.
- A `dfwb.plugins` hook that registers `layers/srm` -> `SRMConv2d` with Deepfake Workbench,
  with no dependency on the framework (this package installs, imports and runs with nothing
  but `torch`).
