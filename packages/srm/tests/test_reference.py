"""Reference tests for srm: the kernel banks against transcribed source tables.

The expected kernels live in ``_tables.py``. They were transcribed by hand
from Fridrich & Kodovsky 2012 (Fig. 2) and the authors' SRM.m, and from Zhou
et al. 2018 (Fig. 4) and the authors' RGB-N code; that module's docstring has
the exact sections, URLs and commits. Any change to a kernel must change a
table too, so it is always a deliberate, reviewed diff.
"""

import dataclasses
import math
import re
from collections import Counter

import pytest
import torch
import torch.nn.functional as F
from _tables import ALL_TABLES, RGBN3_TABLES, SRM30_TABLES, taps_of
from dfwb_torch_srm.kernels import BANKS, KERNELS, KernelInfo, srm_kernels

_KEBAB = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")


def _expected(names: list[str]) -> torch.Tensor:
    return torch.tensor([taps_of(ALL_TABLES[name]) for name in names], dtype=torch.float64)


# --- every kernel -----------------------------------------------------------


@pytest.mark.parametrize("name", list(ALL_TABLES))
def test_every_kernel_matches_its_transcribed_table(name: str) -> None:
    info = KERNELS[name]
    assert info.name == name
    # Exact equality: both sides are the correctly rounded quotient of the
    # same two integers, so there is nothing to tolerate.
    assert info.taps == taps_of(ALL_TABLES[name])
    assert torch.equal(srm_kernels((name,), dtype=torch.float64), _expected([name]))


def test_every_kernel_has_a_transcribed_table() -> None:
    assert list(KERNELS) == list(ALL_TABLES)


@pytest.mark.parametrize("name", list(ALL_TABLES))
def test_every_kernel_is_zero_sum(name: str) -> None:
    _divisor, grid = ALL_TABLES[name]
    assert sum(sum(row) for row in grid) == 0  # the transcription itself
    assert abs(math.fsum(tap for row in KERNELS[name].taps for tap in row)) < 1e-12
    assert abs(srm_kernels((name,), dtype=torch.float64).sum().item()) < 1e-12


def test_every_kernel_is_a_5x5_named_and_cited_kernel_info() -> None:
    for name, info in KERNELS.items():
        assert isinstance(info, KernelInfo)
        assert _KEBAB.match(name), name
        assert len(info.taps) == 5
        assert all(len(row) == 5 for row in info.taps), name
        assert info.source.strip(), name
        assert info.normalisation.strip(), name
    for name in SRM30_TABLES:
        assert "Fridrich" in KERNELS[name].source, name
    for name in BANKS["rgbn3"]:
        assert "Zhou" in KERNELS[name].source, name


def test_taps_follow_the_cross_correlation_convention() -> None:
    # Tap (r, c) multiplies X[i + r - 2, j + c - 2], as in F.conv2d.
    x = torch.randn(1, 1, 9, 9, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
    first = F.conv2d(x, srm_kernels(("first-order-h",), dtype=torch.float64)[:, None])
    assert torch.allclose(first[0, 0], x[0, 0, 2:-2, 3:-1] - x[0, 0, 2:-2, 2:-2])
    down = F.conv2d(x, srm_kernels(("first-order-v",), dtype=torch.float64)[:, None])
    assert torch.allclose(down[0, 0], x[0, 0, 3:-1, 2:-2] - x[0, 0, 2:-2, 2:-2])
    third = F.conv2d(x, srm_kernels(("third-order-h",), dtype=torch.float64)[:, None])
    row = x[0, 0, 2:-2]
    reference = (-row[:, 4:] + 3 * row[:, 3:-1] - 3 * row[:, 2:-2] + row[:, 1:-3]) / 3
    assert torch.allclose(third[0, 0], reference)


# --- banks ------------------------------------------------------------------


def test_srm30_has_30_distinct_kernels() -> None:
    names = BANKS["srm30"]
    assert len(names) == 30
    assert len(set(names)) == 30
    bank = srm_kernels("srm30", dtype=torch.float64)
    assert bank.shape == (30, 5, 5)
    assert torch.unique(bank.reshape(30, 25), dim=0).shape[0] == 30
    # Fridrich & Kodovsky 2012 Fig. 2 / SRM.m: 8 first-order directions,
    # 4 second-order axes, 8 third-order directions, SQUARE 3x3, EDGE 3x3 in
    # 4 rotations, SQUARE 5x5 and EDGE 5x5 in 4 rotations.
    family = re.compile(r"^(first-order|second-order|third-order|square3|edge3|square5|edge5)")
    counts = Counter(m.group(1) for m in map(family.match, names) if m)
    assert counts == {
        "first-order": 8,
        "second-order": 4,
        "third-order": 8,
        "square3": 1,
        "edge3": 4,
        "square5": 1,
        "edge5": 4,
    }


def test_srm30_order_is_pinned() -> None:
    assert BANKS["srm30"] == tuple(SRM30_TABLES)


def test_srm30_centre_tap_is_minus_one() -> None:
    # T6: each SRM residual R = Xhat - c X is divided by c, so the centre is -1.
    for name in BANKS["srm30"]:
        assert KERNELS[name].taps[2][2] == -1.0, name


def test_rgbn3_matches_zhou_2018() -> None:
    expected = torch.tensor([taps_of(t) for t in RGBN3_TABLES.values()], dtype=torch.float64)
    assert torch.equal(srm_kernels("rgbn3", dtype=torch.float64), expected)
    assert BANKS["rgbn3"] == ("square5", "square3", "second-order-h")


def test_banks_only_name_known_kernels() -> None:
    assert set(BANKS) == {"srm30", "rgbn3", "square3", "square5", "laplacian", "sobel"}
    for bank, names in BANKS.items():
        assert isinstance(names, tuple), bank
        assert names, bank
        assert set(names) <= set(KERNELS), bank
    assert BANKS["square3"] == ("square3",)
    assert BANKS["square5"] == ("square5",)
    assert BANKS["laplacian"] == ("laplacian",)
    assert BANKS["sobel"] == ("sobel-x", "sobel-y")


def test_registries_are_read_only() -> None:
    with pytest.raises(TypeError):
        KERNELS["new"] = KERNELS["square3"]  # type: ignore[index]
    with pytest.raises(TypeError):
        BANKS["new"] = ("square3",)  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        KERNELS["square3"].name = "renamed"  # type: ignore[misc]


# --- srm_kernels ------------------------------------------------------------


def test_names_select_in_the_given_order() -> None:
    picked = srm_kernels(("sobel-y", "sobel-x", "sobel-y"), dtype=torch.float64)
    assert torch.equal(picked, _expected(["sobel-y", "sobel-x", "sobel-y"]))
    assert torch.equal(srm_kernels(["laplacian"], dtype=torch.float64), _expected(["laplacian"]))


@pytest.mark.parametrize(
    ("query", "suggestion"),
    [
        ("srm03", "'srm30'"),
        ("rgbn", "'rgbn3'"),
        ("sobell", "'sobel'"),
        (("first-order-hh",), "'first-order-h'"),
        (("sobel-x", "sobelx"), "'sobel-x'"),
        (("square-5",), "'square5'"),
    ],
)
def test_unknown_name_suggests(query: str | tuple[str, ...], suggestion: str) -> None:
    with pytest.raises(KeyError, match=f"did you mean.*{re.escape(suggestion)}"):
        srm_kernels(query)


def test_unknown_name_without_a_close_match_lists_the_known_names() -> None:
    known_banks = "known banks: laplacian, rgbn3, sobel, square3, square5, srm30"
    with pytest.raises(KeyError, match=known_banks):
        srm_kernels("zzzz")
    with pytest.raises(KeyError, match=r"known kernels: .*first-order-h"):
        srm_kernels(("zzzz",))


def test_a_kernel_name_is_not_a_bank() -> None:
    with pytest.raises(KeyError, match=re.escape("pass it in a sequence: ('first-order-h',)")):
        srm_kernels("first-order-h")


def test_an_empty_selection_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one kernel"):
        srm_kernels(())


def test_dtype_and_device_arguments() -> None:
    default = srm_kernels("srm30")
    assert default.shape == (30, 5, 5)
    assert default.dtype == torch.get_default_dtype()
    assert default.device == torch.device("cpu")
    reference = srm_kernels("srm30", dtype=torch.float64)
    for dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        bank = srm_kernels("srm30", dtype=dtype)
        assert bank.dtype == dtype
        # Rounded once from the exact float64 taps, not via an intermediate.
        assert torch.equal(bank, reference.to(dtype))
    assert srm_kernels("sobel", device="meta").device.type == "meta"
    assert srm_kernels("sobel", device=torch.device("cpu")).device == torch.device("cpu")
    with pytest.raises(TypeError, match="floating-point dtype"):
        srm_kernels("srm30", dtype=torch.int64)
    with pytest.raises(TypeError, match="floating-point dtype"):
        srm_kernels("srm30", dtype=torch.complex64)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_dtype_and_device_arguments_cuda() -> None:
    bank = srm_kernels("rgbn3", dtype=torch.float32, device="cuda")
    assert bank.device.type == "cuda"
    assert torch.equal(bank.cpu(), srm_kernels("rgbn3", dtype=torch.float32))


def test_each_call_returns_a_fresh_tensor() -> None:
    first = srm_kernels("square3")
    first.zero_()
    assert srm_kernels("square3").abs().sum() > 0
