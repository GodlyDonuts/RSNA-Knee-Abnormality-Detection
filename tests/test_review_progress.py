import json
from pathlib import Path

from rsna.labels import file_sha256, source_reports_fingerprint
from rsna.review_progress import collect


def review_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    inputs, root, rubric = tmp_path / "inputs", tmp_path / "reviews", tmp_path / "rubric.md"
    inputs.mkdir()
    rubric.write_text("Frozen fixture rubric")
    source = [{"study_uid": "study-a", "report": "A report with no target information."}]
    input_file = inputs / "batch-000.json"
    input_file.write_text(json.dumps(source))
    previous = None
    for stage, folder in (("draft", "drafts"), ("reviewed", "reviewed")):
        (root / folder).mkdir(parents=True)
        (root / "provenance").mkdir(exist_ok=True)
        artifact = root / folder / f"batch-000-{stage}.jsonl"
        annotation = {
            "study_uid": "study-a",
            "states": ["not_mentioned"] * 12,
            "evidence": [""] * 12,
            "notes": "Fixture",
        }
        artifact.write_text(json.dumps(annotation) + "\n")
        meta = {
            "stage": stage,
            "agent_id": stage,
            "session_id": stage,
            "workflow_run_id": "test",
            "notes": "Fixture pass",
            "rubric_version": "test-v1",
            "sha256": file_sha256(artifact),
            "n_studies": 1,
            "study_uids": ["study-a"],
            "rubric_sha256": file_sha256(rubric),
            "source_reports_sha256": source_reports_fingerprint({"study-a": source[0]["report"]}),
            "source_sha256": file_sha256(input_file),
            "parent_artifact_sha256": previous,
        }
        previous = meta["sha256"]
        (root / "provenance" / f"batch-000-{stage}.json").write_text(json.dumps(meta))
    return inputs, root, rubric


def test_collect_validates_complete_review_without_claiming_image_readiness(tmp_path: Path) -> None:
    inputs, root, rubric = review_fixture(tmp_path)
    result = collect(inputs, root, rubric)
    assert result["complete"]
    assert not result["training_ready"]
    provenance = json.loads((root / "provenance.json").read_text())
    assert provenance["batch-000"]["reviewed"]["sha256"]


def test_collect_rejects_self_review_even_with_matching_content_hashes(tmp_path: Path) -> None:
    inputs, root, rubric = review_fixture(tmp_path)
    sidecar = root / "provenance/batch-000-reviewed.json"
    meta = json.loads(sidecar.read_text())
    meta["agent_id"] = "draft"
    sidecar.write_text(json.dumps(meta))
    result = collect(inputs, root, rubric)
    assert not result["complete"]
    assert result["counts"]["reviewed"] == {"invalid": 1}
    assert "independent" in result["batches"][0]["reviewed"]["error"]


def test_collect_preserves_incomplete_artifact_and_blocks_completion(tmp_path: Path) -> None:
    inputs, root, rubric = review_fixture(tmp_path)
    (root / "provenance/batch-000-reviewed.json").unlink()
    result = collect(inputs, root, rubric)
    assert not result["complete"]
    assert (root / "reviewed/batch-000-reviewed.jsonl").exists()
