import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsna.dicom import decode_slice, make_volume, order_slices  # noqa: E402
from tools.research.public_dinov2_input_reference import (  # noqa: E402
    compare,
    contract,
    provenance,
    synthetic,
)

sys.path.insert(0, str(ROOT / "scripts" / "newton"))
import compare_public_dinov2_inputs as runner  # noqa: E402


def sop(path):
    return str(pydicom.dcmread(path, stop_before_pixels=True).SOPInstanceUID)


def notebook(*cells):
    return json.dumps({"cells": [{"cell_type": t, "source": s.splitlines(True)} for t, s in cells]})


def test_exact_source_gate_refuses_other_bytes():
    with pytest.raises(provenance.SourceMismatch, match="observed 4B"):
        provenance.verify_source(b"code")
    with pytest.raises(provenance.SourceMismatch):
        provenance.load_reference(b"x" * provenance.SOURCE_BYTES)


def test_review_pending_refuses_before_any_extraction(monkeypatch):
    review = provenance.load_review()
    assert review["status"] == "pending_source_handoff" and review["nodes"] == []
    monkeypatch.setattr(provenance, "verified_code", lambda raw: "def f():\n    return 1\n")
    with pytest.raises(provenance.SourceReviewPending):
        provenance.load_reference(b"", review)


def test_cell_join_matches_original_collector():
    source = notebook(("markdown", "# t\n"), ("code", "a = 1\nb = 2"))
    assert provenance.notebook_code(source) == "# t\n\n\na = 1\nb = 2"


def test_extract_nodes_is_exact_last_definition_and_never_runs_top_level():
    source = notebook(
        ("code", "!pip install x\nimport numpy as np\nraise SystemExit('top level')\n"),
        ("code", "SIDE = 1\ndef render(x):\n    return x * SIDE\n"),
        ("code", "SIDE = 336\n"),
    )
    text, digest = provenance.extract_nodes(source, ["SIDE", "render"])
    assert text == "def render(x):\n    return x * SIDE\n\nSIDE = 336"
    assert digest == provenance.sha256(text.encode())
    namespace = provenance.compile_nodes(text)
    assert namespace["render"](2) == 672
    with pytest.raises(ValueError, match="absent"):
        provenance.extract_nodes(source, ["missing"])


@pytest.mark.parametrize(
    "body",
    [
        "def f():\n    import pickle\n",
        "def f(p):\n    return open(p)\n",
        "def f(t):\n    return t.load('x')\n",
        "def f():\n    return 'train_label_coordinates.csv'\n",
        "def f():\n    return 'https://example.com'\n",
        "def f():\n    import requests\n",
    ],
)
def test_reviewed_nodes_refuse_labels_files_network_and_pickle(body):
    with pytest.raises(ValueError):
        provenance.extract_nodes(notebook(("code", body)), ["f"])


def test_compiled_namespace_refuses_unlisted_imports():
    namespace = provenance.compile_nodes("def f():\n    import math\n    return math.pi\n")
    assert namespace["f"]() == pytest.approx(3.14159, abs=1e-5)
    with pytest.raises(ValueError, match="__import__"):
        provenance.compile_nodes("def g(name):\n    return __import__(name)\n")
    with pytest.raises(ImportError, match="socket"):
        provenance._guarded_import("socket")


def test_recipe_is_checkpoint_bound_and_unverified_fields_listed():
    assert contract.RECIPE["image_side"].value == 336
    assert contract.RECIPE["slots"].value == 6
    assert contract.RECIPE["channels_per_image"].value == 3
    assert contract.RECIPE["field_of_view_mm"].value == 130.0
    assert contract.RECIPE["selected_slices"].value == 12
    assert contract.RECIPE["slice_band"].value == (0.2, 0.8)
    assert not any(f.source_verified for f in contract.RECIPE.values())
    assert "inference window aggregation" in contract.UNBOUND
    images = torch.zeros(6, 4, 3, 336, 336)
    mask = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)
    assert contract.validate_rendering(images, mask)["occupied_slots"] == 3
    with pytest.raises(ValueError):
        contract.validate_rendering(torch.zeros(6, 3, 224, 224), mask)
    with pytest.raises(ValueError):
        contract.validate_rendering(images.fill_(float("nan")), mask)


@pytest.mark.parametrize("case", sorted(synthetic.CASES))
def test_synthetic_studies_decode_with_known_geometry(tmp_path, case):
    study = synthetic.write_case(tmp_path, case)
    specs = synthetic.CASES[case]
    series_dirs = sorted(study.iterdir(), key=lambda p: int(p.name.rsplit(".", 1)[1]))
    assert len(series_dirs) == len(specs)
    for spec, folder in zip(specs, series_dirs):
        files = sorted(folder.glob("*.dcm"))
        assert len(files) == spec.slices
        file_order = [
            int(pydicom.dcmread(f, stop_before_pixels=True).InstanceNumber) for f in files
        ]
        assert file_order not in (sorted(file_order), sorted(file_order, reverse=True))
        header = pydicom.dcmread(files[0], stop_before_pixels=True)
        assert header.ImageLaterality == spec.laterality
        assert [float(v) for v in header.PixelSpacing] == list(spec.spacing)
        slices = [decode_slice(f.read_bytes(), study.name, folder.name, sop(f)) for f in files]
        ordered, method, _ = order_slices(slices)
        instances = [s.instance for s in ordered]
        if spec.duplicate_positions or spec.constant:
            continue
        assert method.startswith("geometry") or method.startswith("patient")
        assert instances in (sorted(instances), sorted(instances, reverse=True))
        volume, meta = make_volume(ordered, 64)
        assert volume.shape == (spec.slices, 64, 64)
        field = compare.physical_field_mm((spec.rows, spec.columns), spec.spacing)
        assert field == pytest.approx(
            max(spec.rows * spec.spacing[0], spec.columns * spec.spacing[1])
        )


def test_constant_series_is_a_recorded_refusal_in_current_inputs(tmp_path):
    study = synthetic.write_case(tmp_path, "duplicates_and_constant")
    folder = next(p for p in study.iterdir() if p.name.endswith(".2"))
    slices = [
        decode_slice(f.read_bytes(), study.name, folder.name, sop(f))
        for f in sorted(folder.glob("*.dcm"))
    ]
    with pytest.raises(ValueError):
        make_volume(order_slices(slices)[0], 64)


def test_left_and_right_positions_mirror_and_oblique_basis_is_orthonormal():
    for degrees in (0.0, 12.0, -8.0):
        basis = synthetic.orientation("Sagittal", degrees)
        row, col = basis[:3], basis[3:]
        assert np.linalg.norm(row) == pytest.approx(1) and np.dot(row, col) == pytest.approx(0)
    assert synthetic.orientation("Sagittal", 12.0)[0] != pytest.approx(0)


def test_comparison_reports_tensor_mask_and_order_separately():
    a = {
        "images": torch.ones(6, 2, 3, 4, 4),
        "slot_mask": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool),
        "slice_order": [0, 1],
    }
    b = {**a, "images": a["images"] + 1e-4, "slice_order": [1, 0]}
    result = compare.compare_renderings(a, b, atol=1e-3)
    assert result["tensor_exact"] is False and result["tensor_within_tolerance"] is True
    assert result["tensor_max_abs"] == pytest.approx(1e-4, rel=1e-3)
    assert result["slot_mask_equal"] is True and result["slice_order_equal"] is False
    c = {**a, "images": torch.ones(6, 2, 3, 5, 5), "slot_mask": torch.zeros(6, dtype=torch.bool)}
    result = compare.compare_renderings(a, c)
    assert result["shape_equal"] is False and result["slot_mask_equal"] is False
    assert "tensor_exact" not in result


def test_summaries_are_finite_and_sanitizer_refuses_identifiers():
    summary = compare.summarize([1.0, float("nan"), 3.0])
    assert summary == {
        "n": 3,
        "nonfinite": 1,
        "min": 1.0,
        "q25": 1.5,
        "median": 2.0,
        "q75": 2.5,
        "max": 3.0,
    }
    assert compare.sanitize({"roster_size": 2, "refusals": {"ValueError": 1}})
    with pytest.raises(ValueError, match="Unsanitized"):
        compare.sanitize({"study_uids": []})
    with pytest.raises(ValueError, match="UID"):
        compare.sanitize({"order": {"x": "1.2.826.0.1.3680043"}})


def test_roster_is_bounded_deterministic_and_outcome_free():
    uids = pd.Series([f"1.2.3.{i}" for i in range(100)])
    roster = runner.select_roster(uids, 32)
    assert len(roster) == 32 == len(set(roster))
    assert roster == runner.select_roster(uids.sample(frac=1, random_state=4), 32)
    with pytest.raises(ValueError):
        runner.select_roster(uids, runner.MAX_ROSTER + 1)


def test_runner_refuses_off_owning_node(monkeypatch):
    monkeypatch.setattr(runner.os, "uname", lambda: type("U", (), {"nodename": "devin"})())
    with pytest.raises(ValueError, match="owning-node"):
        runner.run(None)
