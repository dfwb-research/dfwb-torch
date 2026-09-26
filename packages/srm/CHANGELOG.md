# Changelog

All notable changes to `dfwb-torch-srm` are documented here.

## [Unreleased]

### Changed

- **`SRMConv2d.__init__`: every argument after `in_channels` is now keyword-only**, matching
  `srm_conv2d`'s own style (e.g. `SRMConv2d(3, mode="gray")`, not `SRMConv2d(3, "srm30", "gray")`).
- Fixed (`trainable=False`) kernels: the `weight` buffer is now built once at construction
  **in float64**, straight from `bank`'s exact taps, regardless of `torch.get_default_dtype()`.
  `forward` casts this buffer to the input's dtype/device on every call (about 750 numbers, so
  the cost is negligible); it caches nothing and mutates no module state. `trainable=True` is
  unchanged. `weight` is the sole source of truth again: editing the buffer in place, or
  loading a different one with `load_state_dict` (including a checkpoint saved by 0.1.0rc1,
  whose buffer was float32 -- the copy casts to the destination's own dtype), changes what
  `forward` computes.
- `srm_conv2d`'s module-internal weight-override hook (used by `SRMConv2d.forward` to share
  its implementation) is no longer part of the public `srm_conv2d` signature.
- `TLU.__init__`'s `threshold` must now be strictly positive (`> 0`), not merely non-negative.
  The stateless `tlu(x, threshold)` function is unchanged and still does not validate its
  `threshold`; its docstring now says so explicitly and cross-references `TLU`, whose
  `threshold` is always in 0-255 residual units regardless of `tlu`'s raw, unscaled one.

### Fixed

- `SRMConv2d.forward` now rejects an input whose channel count does not match `in_channels`
  with a clear `ValueError`, instead of `mode="depthwise"`/`"sum"` silently computing an
  output whose channel count no longer matches `out_channels`.
- `srm_conv2d`/`SRMConv2d.forward` now raise `TypeError` for a non-floating (integer, boolean
  or complex) input, naming the dtype and suggesting `x.float()`. Previously, fixed kernels
  raised a confusing internal error naming `srm_kernels`, not the call actually made, and
  trainable kernels raised nothing at all: they silently computed wrong values in integer
  arithmetic.
- `srm_conv2d`/`SRMConv2d` now reject an empty spatial dimension (`H == 0` or `W == 0`) with
  the package's own `ValueError`, the same as the too-small `padding="valid"` case, instead of
  an opaque `RuntimeError` from deep inside `torch.nn.functional`.
- `SRMConv2d.__init__` now rejects `in_channels < 1` with a `ValueError`, instead of silently
  building a layer with a negative or zero `out_channels`.
- `forward` no longer caches a resolved weight per `(dtype, device)`. That cache mutated module
  state on every new dtype/device a call was made with, which broke every one of the following
  under the original 0.1.0rc1's own design intent, and is now fixed: a call inside
  `torch.inference_mode()` poisoned a later, ordinarily grad-tracked call (`gray`/`per-channel`
  modes) with `RuntimeError: Inference tensors cannot be saved for backward`; `torch.jit.trace`
  on a fresh module failed its own sanity check because the traced graph differed between its
  two calls; a call under `FakeTensorMode` left a `FakeTensor` in the cache that a later, plain
  eager call then silently returned; and editing `weight` in place, or loading a different one,
  had no effect on the output at all, since a `depthwise`/`sum`/`gray` layer already warmed up
  for that dtype/device kept reusing its stale cached tensor.

### Documentation

- Completed the `Raises` sections of `srm_conv2d` and `SRMConv2d.__init__`/`forward` to also
  name the (already-correct) `KeyError` for an unknown bank name and `TypeError` for a
  non-string kernel name.
- Added a "Releasing a package" section to the root `README.md`, listing the tag-day PR steps
  that must all be true of `release.yml`.

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
