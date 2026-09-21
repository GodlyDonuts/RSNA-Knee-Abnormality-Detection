# Reviewed auxiliary series metadata

Use `python -m rsna.series_metadata` to export explicitly reviewed anatomical-plane
corrections into a separate series table. The official CSV, DICOM geometry and
descriptions, cached pixels, and clinical target annotations remain unchanged.
The exporter applies exact study/series decisions; it never infers an override
from an IOP vector, description token, or image.

The 045:14 technical audit illustrates why this distinction matters. Both
independent native-image assessments found that six sequence descriptions match
local knee anatomy, while the official anatomical-plane categories follow
different patient-coordinate planes. All 439 source slices passed integrity and
decode checks, and all ten cached volumes reproduced exactly. The discrepancy
affects auxiliary plane features and plane-aware series selection. It does not
establish corrupted pixels or a general rule for other studies. The specialized
oblique series remains unchanged.

## Decision contract

The required decision JSON has this structure (values below are schematic):

```json
{
  "schema_version": 1,
  "status": "accepted",
  "decision_id": "independently-reviewed-plane-case",
  "source_series_sha256": "<SHA256 of the exact original series CSV>",
  "assessor": "reader identity",
  "reviewer_ids": ["reader identity", "independent reviewer identity"],
  "evidence": [
    {"path": "initiative/audits/case/assessment.json", "sha256": "<exact SHA256>"}
  ],
  "overrides": [
    {
      "study_uid": "<exact StudyInstanceUID>",
      "series_uid": "<exact SeriesInstanceUID>",
      "original_plane": "Coronal",
      "reviewed_plane": "Sagittal"
    }
  ]
}
```

For nonempty overrides, both reviewer identities must be nonempty and distinct,
and the decision assessor must be one of them. Evidence paths are
relative to `--artifact-root`, must remain within it, and must match their exact
hashes. This validates the preserved decision and its evidence bindings; it is
not an automated anatomical review. Keep the independent observations and
accepted decision in the archive.

An optional `bytes` field in each evidence reference is also verified. The
preserved 045:14 decision uses `source_series_path` and `source_revision`; the
exporter verifies the source path when supplied and records the revision as
provenance. It directly accepts the archived decision without an adapter.

The original CSV must have its five official auxiliary columns, unique series
UIDs, valid study/series UIDs, binary fluid/fat fields, and recognized planes.
Every correction must match one row's study, series, and original plane. Unknown
planes, unchanged before/after values, duplicate/conflicting corrections, pending
decisions, changed sources, changed evidence, and already-exported inputs are
rejected. All registered corrections must apply; none are silently skipped.

An explicitly accepted `"overrides": []` exports a source table with no plane
changes. It requires its own exact source hash, an actual `assessor` identity,
and a nonempty `decision` string explaining the pass-through provenance. It does
not require independent reviewer identities or evidence, and makes no anatomical
review claim. A training decision must not be reused against a different test
table, and unknown test series must not inherit heuristic changes.

## Export and use

```bash
python -m rsna.series_metadata \
  --source initiative/source/train_series.csv \
  --decision initiative/audits/anatomical-plane-045-014/decision.json \
  --artifact-root . \
  --output data/series-reviewed-train
```

The output directory must be new. Validation finishes before output creation.
The exporter writes:

- `series_effective.csv`: original row order and values, with accepted changes
  in `Anatomical_Plane`. `Anatomical_Plane_Original` preserves every source value.
  Corrected rows also carry `Anatomical_Plane_Review_ID` and
  `Anatomical_Plane_Review_SHA256`; unchanged rows have empty review lineage.
- `series_metadata_audit.json`: source and decision hashes, all evidence bindings,
  independent identities, exact registered/applied corrections, counts, and the
  effective CSV's checksum and byte size.

Pass the same exported training CSV to existing `--series` arguments for training,
resume, weak validation and heldout inference. The existing dataset reads the
effective `Anatomical_Plane` column and ignores the additional lineage columns.
The training checkpoint's `series_sha256` binds the export bytes. Resume and
heldout inference reject a changed export. Heldout inference also requires the
same study manifest and cache source revision; see
[inference bindings](inference-bindings.md). For test inference, use a separately
source-bound test export. No loader or pixel-processing code changes are required
by this exporter.

Archive the accepted decision, independent evidence, exported CSV, and audit
together, and record the effective CSV hash in the final release manifest.
This command verifies one explicit decision against one table. It does not by
itself establish that all corpus-wide semantic QA or report-review work is done.
