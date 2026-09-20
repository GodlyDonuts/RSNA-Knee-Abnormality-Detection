from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from rsna.archive import import_artifacts, main, review_index, write_json


def prepare_archive(root: Path, count: int = 2) -> None:
    (root / "source").mkdir()
    (root / "reviews").mkdir()
    pd.DataFrame(
        {
            "StudyInstanceUID": [f"study-{index:03d}" for index in range(count)],
            "Report": ["ACL intact."] * count,
            "ACL": [0] * count,
        }
    ).to_csv(root / "source/train.csv", index=False)
    write_json(root / "reviews/provenance.json", {})
    review_index(root, create_inputs=True)


def artifact(root: Path, phase: str, uids: list[str], quote: str = "ACL intact.") -> dict:
    path = root / f"input-{phase}.jsonl"
    rows = [
        {
            "study_uid": uid,
            "states": ["negative", *(["not_mentioned"] * 11)],
            "evidence": [quote, *([""] * 11)],
            "notes": "",
        }
        for uid in uids
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return {
        "batch": "batch-000",
        "phase": phase,
        "path": str(path),
        "session_id": "devin-" + ("a" if phase == "draft" else "d") * 32,
        "agent_id": "wfar-" + ("b" if phase == "draft" else "e") * 32,
        "workflow_run_id": "wfr-" + "c" * 32,
        "n_studies": len(uids),
        "notes": "Fixture annotation.",
    }


def test_archive_requires_both_passes_and_preserves_original_outputs(tmp_path: Path) -> None:
    prepare_archive(tmp_path)
    uids = ["study-000", "study-001"]
    draft = artifact(tmp_path, "draft", uids)
    import_artifacts(tmp_path, [draft])
    assert review_index(tmp_path)["complete"] is False
    reviewed = artifact(tmp_path, "reviewed", uids)
    import_artifacts(tmp_path, [reviewed])
    assert review_index(tmp_path)["complete"] is True
    import_artifacts(tmp_path, [draft, reviewed])
    altered = artifact(tmp_path, "draft", uids, "Invented quotation")
    with pytest.raises(ValueError, match="Cannot replace archived provenance"):
        import_artifacts(tmp_path, [altered])
    archived = tmp_path / "reviews/drafts/batch-000-draft.jsonl"
    archived.write_text(archived.read_text().replace("negative", "positive"))
    with pytest.raises(ValueError, match="differs from recorded agent output"):
        review_index(tmp_path)


def test_archive_rejects_wrong_batch_and_gold_in_blinded_input(tmp_path: Path) -> None:
    prepare_archive(tmp_path, 51)
    inputs = tmp_path / "reviews/inputs/batch-000.json"
    rows = json.loads(inputs.read_text())
    assert len(rows) == 50
    assert set(rows[0]) == {"study_uid", "report"}
    uids = [row["study_uid"] for row in rows]
    uids[0] = "study-050"
    import_artifacts(tmp_path, [artifact(tmp_path, "draft", uids)])
    index = review_index(tmp_path)
    assert index["counts"]["draft"]["invalid"] == 1
    assert index["complete"] is False
    rows[0]["ACL"] = 0
    write_json(inputs, rows)
    with pytest.raises(ValueError, match="Blinded input differs"):
        review_index(tmp_path)


def test_archive_retains_invalid_quotes_but_blocks_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare_archive(tmp_path)
    for phase in ("draft", "reviewed"):
        import_artifacts(
            tmp_path, [artifact(tmp_path, phase, ["study-000", "study-001"], "Fabricated evidence")]
        )
    monkeypatch.setattr(
        sys, "argv", ["archive", "--root", str(tmp_path), "--refresh", "--require-complete"]
    )
    with pytest.raises(ValueError, match="Both report-review stages"):
        main()
    assert review_index(tmp_path)["counts"]["reviewed"]["invalid"] == 1
    monkeypatch.setattr(sys, "argv", ["archive", "--root", str(tmp_path)])
    main()
    (tmp_path / "unrecorded.txt").write_text("unrecorded artifact")
    with pytest.raises(ValueError, match="checksums or membership differ"):
        main()
