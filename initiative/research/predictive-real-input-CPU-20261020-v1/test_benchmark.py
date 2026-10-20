import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("real_input_benchmark", Path(__file__).with_name("benchmark.py"))
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def contexts():
    return [
        {
            **{key: torch.tensor([1.0, 2.0]) for key in
               ("images", "token_metadata", "region_metadata", "targets")},
            "observed": torch.tensor([True, False]),
            "supports": SimpleNamespace(
                indptr=torch.tensor([0, 2]), regions=torch.tensor([0, 1]),
                log_prior=torch.tensor([-0.7, -0.7]), region_count=2,
            ),
        }
        for _ in range(2)
    ]


def test_accepts_all_context_fields():
    benchmark.equal_contexts(contexts(), contexts(), torch)


@pytest.mark.parametrize("key", ["images", "token_metadata", "region_metadata", "targets", "observed"])
def test_refuses_any_changed_context_field(key):
    left, right = contexts(), contexts()
    right[1][key][0] = 0
    with pytest.raises(ValueError, match="Context tensors differ"):
        benchmark.equal_contexts(left, right, torch)


@pytest.mark.parametrize("key", ["indptr", "regions", "log_prior", "region_count"])
def test_refuses_changed_supports(key):
    left, right = contexts(), contexts()
    if key == "region_count":
        right[1]["supports"].region_count = 3
    else:
        getattr(right[1]["supports"], key)[0] += 1
    with pytest.raises(ValueError, match="Support"):
        benchmark.equal_contexts(left, right, torch)
