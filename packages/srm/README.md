# dfwb-torch-srm

SRM and high-pass residual filters

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
from dfwb_torch_srm import Srm

module = Srm()
x = torch.zeros(1, 3, 8, 8)
y = module(x)
```

## API

| Name | What |
|---|---|
| `Srm` | Placeholder module; replace with the real API for srm. |

## Citation

<!-- Cite the method's source paper here: authors, year, venue, and a DOI or URL. -->

## Licence

MIT. See `LICENSE`.
