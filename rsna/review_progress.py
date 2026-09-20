"""Validate incremental provider-neutral annotation artifacts without promoting them."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from rsna.labels import file_sha256, read_reviews, source_reports_fingerprint


def collect(inputs: Path, root: Path, rubric: Path) -> dict:
    batches = {
        path.stem: json.loads(path.read_text()) for path in sorted(inputs.glob("batch-*.json"))
    }
    if not batches:
        raise ValueError("No blinded source batches")
    reports = {}
    for records in batches.values():
        for record in records:
            if record["study_uid"] in reports:
                raise ValueError("Duplicate source study")
            reports[record["study_uid"]] = record["report"]
    expected = sorted(reports)
    for index, (name, records) in enumerate(batches.items()):
        if (
            name != f"batch-{index:03d}"
            or [r["study_uid"] for r in records] != expected[index * 50 : (index + 1) * 50]
        ):
            raise ValueError("Source batch ordering/membership differs from canonical batches")
    source_hash = source_reports_fingerprint(reports)
    rubric_hash = file_sha256(rubric)
    provenance: dict = {}
    records_out = []
    known: set[Path] = set()
    for name, source in batches.items():
        local_reports = {r["study_uid"]: r["report"] for r in source}
        record_out: dict = {"batch": name, "n_studies": len(source)}
        for stage, folder in (("draft", "drafts"), ("reviewed", "reviewed")):
            artifact = root / folder / f"{name}-{stage}.jsonl"
            sidecar = root / "provenance" / f"{name}-{stage}.json"
            known.update((artifact, sidecar))
            if not artifact.exists() and not sidecar.exists():
                record_out[stage] = {"status": "missing"}
                continue
            try:
                if not artifact.exists() or not sidecar.exists():
                    raise ValueError("Artifact/provenance pair is incomplete")
                meta = json.loads(sidecar.read_text())
                if meta.get("stage") != stage:
                    raise ValueError("Stage identity mismatch")
                for field in (
                    "agent_id",
                    "session_id",
                    "workflow_run_id",
                    "notes",
                    "rubric_version",
                ):
                    if not isinstance(meta.get(field), str) or not meta[field].strip():
                        raise ValueError(f"Missing provenance {field}")
                digest = file_sha256(artifact)
                if meta.get("sha256", meta.get("output_sha256")) != digest:
                    raise ValueError("Output checksum mismatch")
                if "output_sha256" in meta and meta["output_sha256"] != digest:
                    raise ValueError("Conflicting output checksum")
                if meta.get("n_studies") != len(source) or isinstance(meta["n_studies"], bool):
                    raise ValueError("Study count mismatch")
                if meta.get("study_uids") != list(local_reports):
                    raise ValueError("Exact UID list mismatch")
                if (
                    meta.get("rubric_sha256") != rubric_hash
                    or meta.get("source_reports_sha256") != source_hash
                ):
                    raise ValueError("Rubric or blinded source fingerprint mismatch")
                if meta.get("source_sha256", meta.get("input_sha256")) != file_sha256(
                    inputs / f"{name}.json"
                ):
                    raise ValueError("Blinded batch file checksum mismatch")
                read_reviews(artifact.parent, local_reports, f"{name}-{stage}")
                if stage == "reviewed":
                    draft = provenance.get(name, {}).get("draft")
                    if draft is None:
                        raise ValueError("No valid parent draft")
                    if meta.get("parent_artifact_sha256") != draft["sha256"]:
                        raise ValueError("Reviewed parent hash mismatch")
                    if any(meta[field] == draft[field] for field in ("agent_id", "session_id")):
                        raise ValueError("Reader and reviewer must be independent")
                normalized = {**meta, "sha256": digest, "path": str(artifact.relative_to(root))}
                provenance.setdefault(name, {})[stage] = normalized
                record_out[stage] = {"status": "valid", "sha256": digest}
            except (ValueError, KeyError, TypeError) as exc:
                record_out[stage] = {"status": "invalid", "error": str(exc)}
        records_out.append(record_out)
    actual = (
        set((root / "drafts").glob("*.jsonl"))
        | set((root / "reviewed").glob("*.jsonl"))
        | set((root / "provenance").glob("*.json"))
    )
    if actual - known:
        raise ValueError("Unexpected artifact or provenance filename")
    counts = {
        stage: dict(Counter(row[stage]["status"] for row in records_out))
        for stage in ("draft", "reviewed")
    }
    result = {
        "rubric_sha256": rubric_hash,
        "source_reports_sha256": source_hash,
        "studies": len(reports),
        "counts": counts,
        "complete": all(counts[stage].get("valid", 0) == len(batches) for stage in counts),
        "training_ready": False,
        "scope": "Evidence, identity and coverage checks only; label semantics require independent reading",
        "batches": records_out,
    }
    root.mkdir(parents=True, exist_ok=True)
    for filename, data in (("index.json", result), ("provenance.json", provenance)):
        temporary = root / f".{filename}.partial"
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(root / filename)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = collect(args.inputs, args.root, args.rubric)
    print(json.dumps({key: value for key, value in result.items() if key != "batches"}, indent=2))
    if any(stage.get("invalid", 0) for stage in result["counts"].values()) or (
        args.require_complete and not result["complete"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
