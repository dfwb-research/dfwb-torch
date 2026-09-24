# dfwb-torch

Small, clean, **standalone** PyTorch building blocks for media forensics that are useful in
*anyone's* code: a paper, a Kaggle kernel, another framework. Each package here is an
independently versioned, independently released PyPI distribution (`dfwb-torch-<name>`) whose
only runtime dependency is `torch` (plus `numpy` where genuinely needed). None of them depends on
`deepfake-workbench`. A package may optionally self-register with
[Deepfake Workbench](https://github.com/dfwb-research/deepfake-workbench) through the
`dfwb.plugins` entry point, which lets the framework adopt it automatically when both are
installed — but every package installs, imports and runs standalone with nothing but `torch`.

> **Status:** pre-release. Linux is the only supported and tested OS. Python ≥ 3.12.

## Packages

| Package | What | Status | PyPI |
|---|---|---|---|
| [`srm`](packages/srm) | SRM and high-pass residual filter banks (Fridrich & Kodovský 2012; Zhou et al. 2018) for PyTorch | pre-release (`0.1.0rc1`) | [![PyPI](https://img.shields.io/pypi/v/dfwb-torch-srm)](https://pypi.org/project/dfwb-torch-srm/) |

## Run from a clone

```bash
git clone https://github.com/dfwb-research/dfwb-torch && cd dfwb-torch
uv sync
```

Each package also has its own quickstart in its `README.md`, and installs on its own
(`pip install dfwb-torch-srm`) without cloning this monorepo.

## Adding a package

See `scripts/new_package.py` and the template in `template/`. A new package needs a concrete user,
no well-maintained equivalent, and a reviewed API in its plan before it lands here.

## Licence

MIT. See `LICENSE`. Each package's licence is also MIT and is packaged with its own distribution.
