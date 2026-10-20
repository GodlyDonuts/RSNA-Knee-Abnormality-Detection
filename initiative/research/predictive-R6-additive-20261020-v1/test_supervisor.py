"""CPU-only fake-child lifecycle tests; fixtures never represent actual trained acceptance."""

import ast
import copy
import csv
import importlib.util
import io
import json
import subprocess
import time
from pathlib import Path

import pytest

D = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


s = load("test_additive_supervisor", D / "supervisor.py")
compiler = load("test_additive_compiler", D / "compile.py")
fusion = load("test_additive_fusion", D / "fusion.py")


def prediction(rows):
    text = io.StringIO()
    writer = csv.writer(text, lineterminator="\n")
    writer.writerow(fusion.COLUMNS)
    for uid, value in rows:
        writer.writerow([uid, *([value] * 12)])
    return text.getvalue().encode()


@pytest.fixture
def case(tmp_path, monkeypatch):
    roster = ["1.2.3", "1.2.4", "1.2.5"]
    roster_path = tmp_path / "test.csv"
    roster_path.write_text("StudyInstanceUID\n" + "\n".join(roster) + "\n")
    roster_hash = s.digest(s.canonical(roster))
    c_csv = prediction(zip(roster, (0.8, 0.2, 0.5), strict=True))
    r_csv = prediction(zip(roster, (0.1, 0.7, 0.4), strict=True))
    monkeypatch.setattr(s, "PUBLIC_ROSTER", roster_hash)
    monkeypatch.setattr(s, "R6_PUBLIC_CSV", s.digest(r_csv))
    runtime = {"torch": "fixture-torch", "timm": "fixture-timm", "pydicom": "fixture-pydicom",
               "pinned_Docker": "fixture-docker"}
    serving = {
        "serving_worker_sha256": "1" * 64, "serving_contract_sha256": "2" * 64,
        "serving_helpers_sha256": {"fixture.py": "3" * 64},
        "model_dataset_ref": "fixture/never-real",
    }
    c_proof = {
        "status": "actual_predictive_exact_test_roster_submission_created",
        "mode": "competition", "test_roster_sha256": roster_hash,
        "actual_test_studies": 3, "submission_csv_absent": False,
        "GPU_names": ["Tesla T4", "Tesla T4"],
        "CSV_bytes": len(c_csv), "CSV_sha256": s.digest(c_csv),
        "models": [{
            "arm": "C", "trained_export_sha256": "a" * 64, "candidate_sha256": "b" * 64,
            "precision_accepted": True, "memory_accepted": True,
        }], **runtime, **serving,
    }
    r_proof = {
        "status": "actual_exact8cell_reference_script_MRI_and_final_reader_accepted",
        "all8code_cells_sha256": s.R6_CELLS, "original_notebook_sha256": s.R6_NOTEBOOK,
        "completed_original_cells": list(range(8)), "rows": 3, "columns": 13,
        "R6_fallbacks": 0, "R6_bundle_id": "fe055406fad6a12f",
        "full_final_reader_output_exists": True, "original_public_example_exact_CSV_match": True,
        "small_output_proofs": {
            name: {"bytes": len(r_csv), "sha256": s.digest(r_csv)} for name in s.R6_OUTPUTS
        },
    }

    def sources(c_proof_override=None, r_proof_override=None, c_extra="", r_extra=""):
        c = (
            "from pathlib import Path\nimport json,time\n"
            "Path('order').write_text('C')\n"
            + c_extra
            + f"\nPath('submission.csv').write_bytes({c_csv!r})\n"
            + f"Path({s.C_ACCEPTED!r}).write_text({json.dumps(c_proof_override or c_proof)!r})\n"
        ).encode()
        r = (
            "from pathlib import Path\nimport json,time\n"
            "assert Path('order').read_text()=='C'\n"
            "assert Path('predictive-R6-additive-v1/C.csv').is_file()\n"
            "assert not Path('submission.csv').exists()\n"
            "Path('order').write_text('CR6')\n"
            + r_extra
            + f"\nfor name in {s.R6_OUTPUTS!r}: Path(name).write_bytes({r_csv!r})\n"
            + f"Path({s.R6_ACCEPTED!r}).write_text({json.dumps(r_proof_override or r_proof)!r})\n"
        ).encode()
        return c, r

    def bound(c, r, mode="candidate"):
        monkeypatch.setattr(s, "R6_DRIVER_SHAS", (s.digest(r),))
        return {
            "mode": mode, "total_watchdog_seconds": s.TOTAL_SECONDS,
            "source_sha256": {"C": s.digest(c), "R6": s.digest(r),
                              "fusion": s.digest((D / "fusion.py").read_bytes())},
            "C_model_sha256": "a" * 64, "C_candidate_sha256": "b" * 64,
            "C_runtime": runtime, "C_serving": serving,
            "standalone_C_public_CSV_sha256": s.digest(c_csv),
        }

    def run(c, r, binding=None):
        return s.run(c, r, (D / "fusion.py").read_bytes(), binding or bound(c, r),
                     working=tmp_path, roster_path=roster_path)

    return dict(
        root=tmp_path, roster=roster, roster_path=roster_path, c_csv=c_csv, r_csv=r_csv,
        c_proof=c_proof, r_proof=r_proof, sources=sources, bound=bound, run=run,
    )


def test_sequential_exact_children_and_complete_fusion(case):
    c, r = case["sources"]()
    receipt = case["run"](c, r, case["bound"](c, r))
    root = case["root"]
    assert (root / "order").read_text() == "CR6"
    assert (root / "predictive-R6-additive-v1/C.csv").read_bytes() == case["c_csv"]
    assert (root / "predictive-R6-additive-v1/R6.csv").read_bytes() == case["r_csv"]
    expected, _ = fusion.fuse_csv(case["r_csv"], case["c_csv"], case["roster"])
    assert (root / "predictive-R6-additive-v1/fusion.csv").read_bytes() == expected
    assert (root / "submission.csv").read_bytes() == expected
    assert receipt["mode"] == "candidate"
    assert receipt["submission_csv_sha256"] == s.digest(expected)
    assert receipt["submission_csv_bytes"] == len(expected)
    assert receipt["entry_gate_accepted"] is False
    assert receipt["stage_order"] == ["C", "R6"]
    assert [row["source_sha256"] for row in receipt["stages"]] == [s.digest(c), s.digest(r)]
    assert [row["CSV_sha256"] for row in receipt["stages"]] == [
        s.digest(case["c_csv"]), s.digest(case["r_csv"]),
    ]
    assert all(row["elapsed_seconds"] > 0 for row in receipt["stages"])
    assert not receipt["runtime_guarantee"]
    assert not any(uid in json.dumps(receipt) for uid in case["roster"])


@pytest.mark.parametrize("name", ("submission.csv", s.C_ACCEPTED, s.R6_ACCEPTED,
                                "predictive-final-serving-v1", "predictive-R6-additive-v1"))
def test_stale_output_or_root_refused_before_child_and_preserved(case, name):
    path = case["root"] / name
    path.write_bytes(b"original-stale-preserved")
    with pytest.raises(ValueError, match="Fresh"):
        case["run"](*case["sources"]())
    assert path.read_bytes() == b"original-stale-preserved"
    assert not (case["root"] / "order").exists()


@pytest.mark.parametrize("field", ("timm", "CSV_sha256"))
def test_c_receipt_mismatch_prevents_r6_and_removes_intermediate(case, field):
    proof = copy.deepcopy(case["c_proof"])
    proof[field] = "wrong"
    with pytest.raises(ValueError):
        case["run"](*case["sources"](c_proof_override=proof))
    assert (case["root"] / "order").read_text() == "C"
    assert not (case["root"] / "submission.csv").exists()
    assert not (case["root"] / s.COMBINED_ACCEPTED).exists()


@pytest.mark.parametrize("field,value", (
    ("R6_fallbacks", 1), ("completed_original_cells", list(range(7))),
    ("original_public_example_exact_CSV_match", False),
))
def test_r6_acceptance_failure_never_leaves_submission(case, field, value):
    proof = copy.deepcopy(case["r_proof"])
    proof[field] = value
    with pytest.raises(ValueError):
        case["run"](*case["sources"](r_proof_override=proof))
    assert not (case["root"] / "submission.csv").exists()
    assert (case["root"] / "predictive-R6-additive-v1/C.csv").is_file()


def test_public_c_byte_equality_is_mandatory(case):
    c, r = case["sources"]()
    binding = case["bound"](c, r)
    binding["standalone_C_public_CSV_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="standalone"):
        case["run"](c, r, binding)
    assert (case["root"] / "order").read_text() == "C"
    assert not (case["root"] / "submission.csv").exists()


def test_child_failure_removes_partial_output_and_never_runs_next(case):
    c, r = case["sources"](
        c_extra="Path('submission.csv').write_text('partial')\nraise SystemExit(7)\n",
    )
    with pytest.raises(ValueError, match="Child execution failed"):
        case["run"](c, r)
    assert (case["root"] / "order").read_text() == "C"
    assert not (case["root"] / "submission.csv").exists()


def test_watchdog_kills_descendant_process_group_and_removes_partial(case, monkeypatch):
    monkeypatch.setattr(s, "TOTAL_SECONDS", 1.0)
    c, r = case["sources"](c_extra=(
        "import subprocess,sys,os\n"
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'])\n"
        "Path('grandchild.pid').write_text(str(p.pid))\n"
        "Path('submission.csv').write_text('partial')\n"
        "time.sleep(60)\n"
    ))
    start = time.monotonic()
    with pytest.raises((TimeoutError, subprocess.TimeoutExpired)):
        case["run"](c, r)
    assert time.monotonic() - start < 5
    assert not (case["root"] / "submission.csv").exists()
    pid = (case["root"] / "grandchild.pid").read_text()
    status = subprocess.run(
        ["ps", "-o", "stat=", "-p", pid], capture_output=True, text=True, check=False,
    ).stdout.strip()
    assert not status or status.startswith("Z")
    assert (case["root"] / "order").read_text() == "C"


def test_watchdog_is_shared_across_children(case, monkeypatch):
    monkeypatch.setattr(s, "TOTAL_SECONDS", 0.7)
    c, r = case["sources"](c_extra="time.sleep(0.45)\n", r_extra="time.sleep(0.45)\n")
    with pytest.raises((TimeoutError, subprocess.TimeoutExpired)):
        case["run"](c, r)
    assert (case["root"] / "order").read_text() == "CR6"
    assert not (case["root"] / "submission.csv").exists()


def test_stage_source_hash_mismatch_refused_before_any_child(case):
    c, r = case["sources"]()
    binding = case["bound"](c, r)
    binding["source_sha256"]["C"] = "0" * 64
    with pytest.raises(ValueError, match="source hash"):
        case["run"](c, r, binding)
    assert not (case["root"] / "order").exists()


def test_official_roster_allows_metadata_and_alternate_image_directory(tmp_path):
    root = tmp_path / "competitions/rsna-knee-abnormality-detection"
    root.mkdir(parents=True)
    (root / "test_images").mkdir()
    path = root / "test.csv"
    path.write_text("Metadata,StudyInstanceUID\nignored,1.2.3\nignored,1.2.4\n")
    assert s._official_roster(tmp_path) == path
    uids, file_hash = s._roster(path)
    assert uids == ["1.2.3", "1.2.4"]
    assert file_hash == s.digest(path.read_bytes())


@pytest.mark.parametrize("text", (
    "StudyInstanceUID,StudyInstanceUID\n1.2.3,1.2.3\n",
    "Metadata\nignored\n", "StudyInstanceUID,Metadata\n1.2.3\n",
    "StudyInstanceUID\n1.2.3\n1.2.3\n",
))
def test_invalid_official_headers_rows_or_duplicates_rejected(tmp_path, text):
    path = tmp_path / "test.csv"
    path.write_text(text)
    with pytest.raises(ValueError):
        s._roster(path)


def test_real_compiler_driver_pins_are_mandatory_and_not_parameters():
    assert compiler.supervisor.R6_DRIVER_SHAS == (
        "affa44363c530eed37878d083707d4fc50dcf1571d8dfcce12fd1c5a7d4bc581",
        "b472091d7a323e0165958ca01e6f023b21db9c3d2b10d3f79ba8962fb83efc5b",
    )
    with pytest.raises(ValueError, match="exact admitted"):
        compiler.build(
            b"arbitrary driver", "", {}, qualification={}, model_dataset_ref="a/b",
            standalone_c_source_sha256="0" * 64, standalone_c_acceptance={}, mode="candidate",
        )


def test_compiler_calls_canonical_c_build_and_requires_standalone_equality(case, monkeypatch):
    c, r = case["sources"]()
    chosen = {"arm": "C", "model": {"sha256": "a" * 64}}
    proof = copy.deepcopy(case["c_proof"])
    proof["models"][0]["candidate_sha256"] = s.digest(s.canonical(chosen))
    calls = []

    def build(parent, exports, **options):
        calls.append((parent, exports, options))
        return c.decode(), {"source_sha256": s.digest(c)}

    monkeypatch.setattr(compiler.supervisor, "R6_DRIVER_SHAS", (s.digest(r),))
    monkeypatch.setattr(compiler.supervisor, "PUBLIC_ROSTER", proof["test_roster_sha256"])
    monkeypatch.setattr(compiler.predictive.contract, "candidate", lambda **entry: chosen)
    monkeypatch.setattr(compiler.predictive, "build", build)
    entry = {"fixture": "never an actual export"}
    kwargs = dict(
        qualification=proof, model_dataset_ref="fixture/never-real",
        standalone_c_source_sha256=s.digest(c), standalone_c_acceptance=proof, mode="candidate",
    )
    source, receipt = compiler.build(r, "fixture parent", entry, **kwargs)
    ast.parse(source)
    assert calls == [("fixture parent", [entry], {
        "mode": "competition", "model_dataset_ref": "fixture/never-real", "qualification": proof,
    })]
    assert receipt["source_sha256"] == s.digest(source.encode())
    assert receipt["binding"]["source_sha256"]["R6"] == s.digest(r)
    assert receipt["binding"]["source_sha256"]["C"] == s.digest(c)
    assert not receipt["combined_runtime_accepted"]
    assert not receipt["actual_combined_GPU_execution"]
    assert not receipt["compilation_grants_entry_authority"]
    assert receipt["actual_public_receipt_and_runtime_entry_gate_required"]
    kwargs["standalone_c_source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="standalone accepted source"):
        compiler.build(r, "fixture parent", entry, **kwargs)


def test_entry_needs_same_candidate_public_measurement_and_bound_hidden_scenario(case, monkeypatch):
    c, r = case["sources"]()
    binding = case["bound"](c, r)
    measurement = case["run"](c, r, binding)
    source_bytes = copy.deepcopy(binding)
    output_before_gate = (case["root"] / "submission.csv").read_bytes()
    monkeypatch.setattr(compiler.supervisor, "PUBLIC_ROSTER", measurement["test_roster_sha256"])
    monkeypatch.setattr(compiler.supervisor, "R6_PUBLIC_CSV", s.digest(case["r_csv"]))
    with pytest.raises(ValueError, match="explicit hidden"):
        compiler.accept_entry(binding, measurement, None)
    decision = {
        "status": "accepted_combined_C_R6_hidden_runtime_scenario",
        "combined_public_receipt_sha256": s.digest(s.canonical(measurement)),
        "source_sha256": binding["source_sha256"], "C_model_sha256": binding["C_model_sha256"],
        "planned_hidden_studies": 1300,
        "scenario_basis": "synthetic CPU fixture only; no real runtime acceptance",
        "hidden_runtime_is_guaranteed": False,
        "C_projected_seconds": 10000, "R6_projected_seconds": 15000,
        "startup_seconds": 600, "tail_margin_seconds": 1800, "projected_total_seconds": 27400,
        "C_projection_evidence_sha256": "c" * 64, "R6_projection_evidence_sha256": "d" * 64,
    }
    gate = compiler.accept_entry(binding, measurement, decision)
    assert gate["status"] == "accepted_same_candidate_public_run_and_combined_runtime_for_entry"
    assert gate["public_fusion_csv_sha256"] == s.digest(output_before_gate)
    assert not gate["recompile_or_second_public_run_required"]
    assert not gate["competition_entry_submitted"] and gate["new_remote_mutations"] == 0
    assert binding == source_bytes
    assert (case["root"] / "submission.csv").read_bytes() == output_before_gate
    for field, value in (
        ("planned_hidden_studies", 3),
        ("startup_seconds", 1),
        ("tail_margin_seconds", 1),
    ):
        reduced = copy.deepcopy(decision)
        reduced[field] = value
        reduced["projected_total_seconds"] = sum(
            reduced[key] for key in (
                "C_projected_seconds", "R6_projected_seconds",
                "startup_seconds", "tail_margin_seconds",
            )
        )
        with pytest.raises(ValueError):
            compiler.accept_entry(binding, measurement, reduced)
    invalid = copy.deepcopy(decision)
    invalid["R6_projected_seconds"] = 21000
    invalid["projected_total_seconds"] = 33400
    with pytest.raises(ValueError, match="nine hours"):
        compiler.accept_entry(binding, measurement, invalid)
    wrong = copy.deepcopy(measurement)
    wrong["stages"][1]["CSV_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        compiler.accept_entry(binding, wrong, decision)
    for field, value in (
        ("mode", "qualification"), ("submission_csv_created", False),
        ("submission_csv_sha256", "0" * 64), ("binding_sha256", "0" * 64),
    ):
        wrong = copy.deepcopy(measurement)
        wrong[field] = value
        with pytest.raises(ValueError):
            compiler.accept_entry(binding, wrong, decision)


@pytest.mark.parametrize("mode", ("qualification", "competition"))
def test_removed_dual_run_modes_refused_before_execution(case, mode):
    c, r = case["sources"]()
    with pytest.raises(ValueError, match="one complete"):
        case["run"](c, r, case["bound"](c, r, mode))
    assert not (case["root"] / "order").exists()
