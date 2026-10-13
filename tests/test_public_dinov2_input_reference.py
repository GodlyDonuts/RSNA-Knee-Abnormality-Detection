import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rsna.dicom import decode_slice, make_volume, order_slices  # noqa: E402
from tools.research.public_dinov2_input_reference import (  # noqa: E402
    compare,
    contract,
    provenance,
    reference,
    synthetic,
)

sys.path.insert(0, str(ROOT / "scripts" / "newton"))
import compare_public_dinov2_inputs as runner  # noqa: E402

RECOVERED = ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]
PUBLIC = ["SAG_FLUID", "COR_FLUID", "AX_FLUID", "SAG_STRUCT", "COR_STRUCT", "AX_STRUCT"]
NATIVE = {"order": "normal", "lat": "centre", "slot_fallback": False, "decode_fill": "nearest"}
LEGACY = {"order": "dominant_axis", "lat": "corner_x", "slot_fallback": True, "decode_fill": "zero"}


def pixel_group(rules=None, slots=RECOVERED):
    group = {**reference.MANIFEST_BOUND_PIXEL_GROUP, "slots": slots}
    return group if rules is None else {**group, "rules": rules}


@pytest.fixture(scope="module")
def source():
    return reference.SOURCE_PATH.read_bytes()


def configured(source, rules=None, slots=RECOVERED):
    messages: list[str] = []
    namespace, receipt = reference.load(source, messages)
    reference.configure(namespace, pixel_group(rules, slots))
    return namespace, receipt


def sop(path):
    return str(pydicom.dcmread(path, stop_before_pixels=True).SOPInstanceUID)


def notebook(*cells):
    return json.dumps({"cells": [{"cell_type": t, "source": s.splitlines(True)} for t, s in cells]})


def header(path):
    return pydicom.dcmread(path, stop_before_pixels=True)


def test_handoff_bytes_reproduce_source_and_extraction_digests(source):
    assert len(source) == provenance.SOURCE_BYTES
    code = provenance.verified_code(source)
    assert provenance.sha256(code.encode()) == provenance.EXTRACTED_SHA256
    handoff = json.loads((reference.SOURCE_PATH.parent / "provenance.json").read_text())
    assert handoff["sha256"] == provenance.SOURCE_SHA256 and handoff["version"] == 15
    with pytest.raises(provenance.SourceMismatch, match="observed 4B"):
        provenance.verify_source(b"code")
    with pytest.raises(provenance.SourceMismatch):
        provenance.load_reference(b"x" * provenance.SOURCE_BYTES)


def test_reviewed_nodes_bind_digest_and_exclude_label_training_paths(source):
    review = provenance.load_review()
    assert review["status"] == "reviewed"
    namespace, receipt = reference.load(source)
    assert receipt["node_text_sha256"] == review["node_text_sha256"]
    assert receipt["notebook_top_level_executed"] is False
    for absent in ("read_labels", "find_label_table", "extract", "main", "infer_from_package"):
        assert absent not in namespace and absent not in review["nodes"]
    assert namespace["CROP_MM"] == 130.0 and namespace["SLICE_BAND"] == (0.2, 0.8)
    tampered = {**review, "node_text_sha256": "0" * 64}
    with pytest.raises(provenance.SourceMismatch, match="digest"):
        provenance.load_reference(source, tampered, reference.adapters())
    with pytest.raises(ValueError, match="adapter_names"):
        provenance.load_reference(source, review, {})


def test_review_pending_refuses_before_any_extraction(monkeypatch):
    review = {**provenance.load_review(), "status": "pending_source_handoff", "nodes": []}
    monkeypatch.setattr(provenance, "verified_code", lambda raw: "")
    monkeypatch.setattr(provenance, "verify_source", lambda raw: "")
    with pytest.raises(provenance.SourceReviewPending):
        provenance.load_reference(b"", review)


def test_reviewed_exceptions_are_static_and_branch_stays_refused(source):
    namespace, _ = configured(source)
    assert set(vars(provenance.MetadataPandas)) & {"read_csv", "DataFrame"} == set()
    with pytest.raises(ImportError, match="json"):
        provenance._guarded_import("json")
    namespace["ORDER_CACHE"] = "order.json"
    with pytest.raises(RuntimeError, match="refuses Path"):
        namespace["build_cache"]({}, {}, {}, "t")
    exceptions = provenance.load_review()["reviewed_exceptions"]
    unreviewed = notebook(("code", "def build_cache():\n    import json\n"))
    with pytest.raises(ValueError, match="json"):
        provenance.extract_nodes(unreviewed, ["build_cache"])
    with pytest.raises(ValueError, match="outside"):
        provenance.extract_nodes(unreviewed, ["build_cache"], {"annotate": exceptions["annotate"]})


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
    assert provenance.compile_nodes(text)["render"](2) == 672
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
        "def f(p):\n    return pd.read_csv(p)\n",
    ],
)
def test_reviewed_nodes_refuse_labels_files_network_and_pickle(body):
    with pytest.raises(ValueError):
        provenance.extract_nodes(notebook(("code", body)), ["f"])


def test_configure_uses_original_member_checks(source):
    namespace, _ = configured(source, slots=PUBLIC)
    assert [s[0] for s in namespace["SLOTS"]] == PUBLIC
    assert (namespace["IMG"], namespace["CACHE_SLICES"], namespace["N_GROUP"]) == (336, 12, 4)
    assert namespace["RULES"] == NATIVE
    with pytest.raises(ValueError, match="neither"):
        reference.configure(namespace, pixel_group(slots=RECOVERED[::-1]))
    with pytest.raises(namespace["WeightsError"], match="cannot reproduce"):
        reference.configure(namespace, pixel_group({"order": "other"}))


def test_recipe_fields_and_contract(source):
    assert contract.RECIPE["field_of_view_mm"].value == 130.0
    assert contract.RECIPE["selected_slices"].value == 12
    assert all(f.source_verified for f in contract.RECIPE.values())
    assert any("rules" in item for item in contract.UNBOUND)
    images = torch.zeros(6, 12, 336, 336, dtype=torch.uint8)
    mask = torch.tensor([1, 1, 0, 0, 1, 0], dtype=torch.bool)
    assert contract.validate_rendering(images, mask)["occupied_slots"] == 3
    with pytest.raises(ValueError):
        contract.validate_rendering(images.float(), mask)
    images[2, 0, 0, 0] = 1
    with pytest.raises(ValueError, match="Absent"):
        contract.validate_rendering(images, mask)


@pytest.mark.parametrize("rules", [None, LEGACY], ids=["native", "legacy"])
def test_callable_renderer_equals_original_batch_path(tmp_path, source, rules):
    namespace, _ = configured(source, rules)
    studies = {name: synthetic.write_case(tmp_path, name) for name in sorted(synthetic.CASES)}
    planes = {}
    for name, study in studies.items():
        planes.update(synthetic.plane_map(study, synthetic.CASES[name]))
    batch = reference.render_original(namespace, list(studies.values()), planes)
    assert batch["studies"] == sorted(s.name for s in studies.values())
    for name, study in studies.items():
        single = reference.render_study(namespace, study, planes)
        contract.validate_rendering(single["images"], single["slot_mask"])
        row = batch["studies"].index(study.name)
        original = {
            "images": torch.from_numpy(batch["cache"][row]),
            "slot_mask": torch.from_numpy(batch["mask"][row]).bool(),
            "slot_series": [
                str(batch["slots"][study.name][s[0]]["SeriesInstanceUID"])
                if s[0] in batch["slots"][study.name]
                else None
                for s in namespace["SLOTS"]
            ],
        }
        result = compare.compare_renderings(original, single)
        assert result["tensor_exact"] and result["slot_mask_equal"], name
        assert result["slot_series_equal"], name
        assert single["laterality"] == batch["lat"][study.name]


def through_plane(path, rules):
    ds = header(path)
    ipp = np.asarray(ds.ImagePositionPatient, float)
    if rules is None:
        iop = np.asarray(ds.ImageOrientationPatient, float)
        return float(ipp @ np.cross(iop[:3], iop[3:]))
    return ipp


@pytest.mark.parametrize("rules", [None, LEGACY], ids=["native", "legacy"])
def test_scrambled_oblique_stacks_are_ordered_from_geometry(tmp_path, source, rules):
    namespace, _ = configured(source, rules)
    for name in ("complete_right", "left_oblique"):
        study = synthetic.write_case(tmp_path, name)
        out = reference.render_study(
            namespace, study, synthetic.plane_map(study, synthetic.CASES[name])
        )
        assert (
            out["geometry_ordered"][0] == out["geometry_ordered"][1] == int(out["slot_mask"].sum())
        )
        for series, files in zip(out["slot_series"], out["slice_order"]):
            if series is None:
                continue
            assert files != sorted(files)
            keys = [through_plane(study / series / f, rules) for f in files]
            if rules is None:
                assert keys == sorted(keys)
            else:
                xyz = np.stack(keys)
                axis = int(np.argmax(np.ptp(xyz, axis=0)))
                assert list(xyz[:, axis]) == sorted(xyz[:, axis])


def test_right_knee_is_mirrored_onto_left_convention(tmp_path, source):
    namespace, _ = configured(source)
    out = {}
    for side in ("left", "right"):
        study = synthetic.write_case(tmp_path, side, synthetic.MIRROR)
        planes = synthetic.plane_map(study, synthetic.MIRROR[side])
        out[side] = reference.render_study(namespace, study, planes)
    assert (out["left"]["laterality"], out["right"]["laterality"]) == ("L", "R")
    assert out["left"]["laterality_source"] == out["right"]["laterality_source"] == "tag"
    left, right = out["left"]["images"], out["right"]["images"]
    assert torch.equal(right[1], torch.flip(left[1], dims=[-1]))  # COR_FLUID_FS: width flip
    assert torch.equal(right[0], torch.flip(left[0], dims=[0]))  # SAG_FLUID_FS: slice reversal
    assert not torch.equal(right[1], left[1])


def test_absent_slots_and_legacy_t1_fallback_quirk(tmp_path, source):
    study = synthetic.write_case(tmp_path, "t1_fallback_untagged")
    planes = synthetic.plane_map(study, synthetic.CASES["t1_fallback_untagged"])
    native = reference.render_study(configured(source)[0], study, planes)
    legacy = reference.render_study(configured(source, LEGACY)[0], study, planes)
    assert native["slot_mask"].tolist() == [False, False, False, True, False, False]
    assert legacy["slot_mask"].tolist() == [False, False, False, True, True, True]
    assert legacy["slot_series"][5] == legacy["slot_series"][3] == f"{study.name}.1"
    assert legacy["slot_series"][4] == f"{study.name}.2"  # ties broken toward more slices
    assert native["images"][~native["slot_mask"]].abs().sum() == 0
    assert native["laterality"] == legacy["laterality"] == "L"
    assert native["laterality_source"] == legacy["laterality_source"] == "geometry"


def test_band_sampling_pads_short_stacks_with_last_index(tmp_path, source):
    namespace, _ = configured(source)
    study = synthetic.write_case(tmp_path, "absent_slots")
    planes = synthetic.plane_map(study, synthetic.CASES["absent_slots"])
    rendered = reference.render_study(namespace, study, planes)
    assert rendered["laterality"] == "R"
    # The fixture is right-sided; undo the original sagittal reversal before
    # checking the sampler's pre-normalization padding order.
    nine = torch.flip(rendered["images"][0], dims=[0])
    # n=9: band [1, 6] -> unique(linspace) = 1..6 then the last index repeats.
    assert not any(torch.equal(nine[k], nine[k + 1]) for k in range(5))
    assert all(torch.equal(nine[5], nine[k]) for k in range(6, 12))
    study = synthetic.write_case(tmp_path, "duplicates_and_constant")
    out = reference.render_study(
        namespace, study, synthetic.plane_map(study, synthetic.CASES["duplicates_and_constant"])
    )
    three = out["images"][2]
    assert not torch.equal(three[0], three[1])
    assert all(torch.equal(three[1], three[k]) for k in range(2, 12))
    assert bool(out["slot_mask"][1]) and out["images"][1].sum() == 0  # constant -> zeros, present


def test_constant_series_is_a_recorded_refusal_in_current_inputs(tmp_path):
    study = synthetic.write_case(tmp_path, "duplicates_and_constant")
    folder = next(p for p in study.iterdir() if p.name.endswith(".2"))
    slices = [
        decode_slice(f.read_bytes(), study.name, folder.name, sop(f))
        for f in sorted(folder.glob("*.dcm"))
    ]
    with pytest.raises(ValueError):
        make_volume(order_slices(slices)[0], 64)


def test_anisotropic_large_field_matches_documented_crop_window_resize(tmp_path, source):
    namespace, _ = configured(source)
    study = synthetic.write_case(tmp_path, "large_field")
    out = reference.render_study(
        namespace, study, synthetic.plane_map(study, synthetic.CASES["large_field"])
    )
    files = out["slice_order"][1]
    folder = study / out["slot_series"][1]
    n = len(files)
    lo, hi = int(0.2 * (n - 1)), int(0.8 * (n - 1))
    idx = np.unique(np.linspace(lo, hi, 12).astype(int))
    vol = np.stack([pydicom.dcmread(folder / files[i]).pixel_array.astype(np.float32) for i in idx])
    half = int(round(130.0 / 1.8)) // 2
    vol = vol[:, 48 - half : 48 + half, 64 - half : 64 + half]
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip((vol - p1) / max(p99 - p1, 1e-6), 0, 1)
    expected = F.interpolate(torch.from_numpy(vol)[None], size=(336, 336), mode="bilinear")
    expected = (expected[0] * 255).round().clamp(0, 255).to(torch.uint8)
    assert out["laterality"] == "R"  # coronal right knee: width flip after resize
    assert torch.equal(out["images"][1], torch.flip(expected, dims=[-1]))
    assert reference.public_field_mm(namespace, 1.8, 96, 128) == pytest.approx(129.6)
    assert reference.public_field_mm(namespace, 0.3, 64, 48) == pytest.approx(19.2)


def test_inference_windows_and_probability_mean(source):
    namespace, _ = configured(source)
    assert namespace["window_starts"](12, 3) == list(range(10))
    images = torch.zeros((6, 12, 8, 8), dtype=torch.uint8)
    images[:, 3:] = 255
    win = reference.windows(namespace, images)
    assert win.shape == (10, 6, 3, 8, 8) and torch.equal(win[4], images[:, 4:7])

    class Model(torch.nn.Module):
        def forward(self, rows, mask, img_size):
            return (12 * (rows.float().mean(dim=(1, 2, 3, 4)) / 255 - 0.5))[:, None].repeat(1, 12)

    model = Model()

    cache = images[None].numpy()
    mask = np.ones((1, 6), np.float32)
    got = namespace["predict_member"](model, cache, mask, np.arange(1), torch.device("cpu"), 8)
    expected = torch.stack([torch.sigmoid(model(w[None], None, 8)) for w in win]).mean(0)
    assert np.allclose(got, expected.numpy(), atol=1e-6)
    logit_pool = torch.sigmoid(torch.stack([model(w[None], None, 8) for w in win]).mean(0))
    assert float((expected - logit_pool).abs().max()) > 0.1


@pytest.mark.parametrize("case", sorted(synthetic.CASES))
def test_synthetic_studies_decode_with_known_geometry(tmp_path, case):
    study = synthetic.write_case(tmp_path, case)
    specs = synthetic.CASES[case]
    series_dirs = sorted(study.iterdir(), key=lambda p: int(p.name.rsplit(".", 1)[1]))
    assert len(series_dirs) == len(specs)
    for spec, folder in zip(specs, series_dirs):
        files = sorted(folder.glob("*.dcm"))
        assert len(files) == spec.slices
        file_order = [int(header(f).InstanceNumber) for f in files]
        assert file_order not in (sorted(file_order), sorted(file_order, reverse=True))
        first = header(files[0])
        assert getattr(first, "ImageLaterality", None) == (
            spec.laterality if spec.laterality_tag else None
        )
        assert [float(v) for v in first.PixelSpacing] == list(spec.spacing)


def test_comparison_reports_tensor_mask_and_order_separately():
    a = {
        "images": torch.ones(6, 2, 3, 4, 4),
        "slot_mask": torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool),
        "slice_order": [0, 1],
    }
    b = {**a, "images": a["images"] + 1e-4, "slice_order": [1, 0]}
    result = compare.compare_renderings(a, b, atol=1e-3)
    assert result["tensor_exact"] is False and result["tensor_within_tolerance"] is True
    assert result["slot_mask_equal"] is True and result["slice_order_equal"] is False
    c = {**a, "images": torch.ones(6, 2, 3, 5, 5), "slot_mask": torch.zeros(6, dtype=torch.bool)}
    result = compare.compare_renderings(a, c)
    assert result["shape_equal"] is False and "tensor_exact" not in result


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


def test_runner_reads_member_pixel_group_and_distinct_channels():
    group = json.dumps(pixel_group(LEGACY))
    manifest = {
        "members": [
            {"file": provenance.CHECKPOINT, "pixel_group": group, "config": {"pool": "cls_mean"}}
        ]
    }
    member = runner.member_pixel_group(json.dumps(manifest).encode())
    assert member["pixel_group"]["rules"] == LEGACY and member["config"] == {"pool": "cls_mean"}
    with pytest.raises(ValueError, match="exactly once"):
        runner.member_pixel_group(json.dumps({"members": manifest["members"] * 2}).encode())
    images = torch.zeros(6, 12, 2, 2, dtype=torch.uint8)
    images[0, :3] = torch.arange(3)[:, None, None]
    assert runner.distinct_channels(images, torch.tensor([1, 1, 0, 0, 0, 0]).bool()) == [4, 1]


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


def test_extraction_preserves_original_comments_quotes_and_decorators(monkeypatch):
    body = '@torch.no_grad()\ndef f(x):\n    # Preserve this source comment.\n    return "original literal", x+1'
    source = notebook(("code", body))
    monkeypatch.setattr(
        provenance.ast,
        "unparse",
        lambda *a: (_ for _ in ()).throw(AssertionError("unparse must not determine provenance")),
    )
    text, digest = provenance.extract_nodes(source, ["f"])
    assert text == body and digest == provenance.sha256(body.encode())
    assert provenance.compile_nodes(text, {"torch": torch})["f"](2) == ("original literal", 3)


@pytest.mark.parametrize("requested,completed", [(0, 0), (32, 0), (2, 3), (65, 1), (True, 1)])
def test_empty_invalid_or_unbounded_actual_comparison_is_refused(requested, completed):
    with pytest.raises(ValueError, match="nonempty"):
        runner.comparison_population(requested, completed)


def test_actual_input_report_identifies_partial_population():
    assert runner.comparison_population(32, 8) == {"requested": 32, "completed": 8, "partial": True}
    assert runner.comparison_population(32, 32)["partial"] is False
    report = {"comparison_population": runner.comparison_population(8, 8)}
    assert compare.sanitize(report) == report


@pytest.mark.parametrize(
    "population",
    [
        {"requested": 8, "completed": 0, "partial": True},
        {"requested": 8, "completed": 8, "partial": False, "study_uid": "not-exported"},
        {"requested": 8, "completed": 8, "partial": True},
    ],
)
def test_comparison_population_export_is_exact_and_nonempty(population):
    with pytest.raises(ValueError, match="population"):
        compare.sanitize({"comparison_population": population})
