# Archive newly completed work

## Agent outputs

1. Inspect the workflow and completed child sessions with the Devin tools.
   Record batch name, phase, `wfar-...` agent ID, `devin-...` session ID, workflow
   ID, `n_studies`, and the complete structured-output `notes`.
2. Download each actual `artifact_url` with `download_attachment`. Do not replace
   files with links or rely on storage URLs remaining available.
3. Build a local import JSON array outside the repo. One entry has this shape:

```json
{
  "batch": "batch-000",
  "phase": "draft",
  "path": "/absolute/download/path/batch-000-draft.jsonl",
  "session_id": "devin-<32 hexadecimal characters>",
  "agent_id": "wfar-<32 hexadecimal characters>",
  "workflow_run_id": "wfr-b7eeba0fae7f4af18b2f5a869214114c",
  "n_studies": 50,
  "notes": "The exact completion notes returned by the agent."
}
```

Use `phase: "reviewed"` for an independent review. Do not infer a review from an
annotator's statement that they checked their own output.

```bash
python -m rsna.archive --refresh --imports /absolute/path/imports.json
python -m rsna.archive
```

The importer copies file bytes unchanged, adds hashes and provenance, and refuses
to overwrite an existing output or its recorded metadata. It creates/validates
the 89 blinded source batches. The generated index tests each artifact against
its exact batch rather than only global UID coverage. Missing stages remain
`not_archived`. Invalid originals remain recorded as `invalid` with an error.
They cannot pass `--require-complete`.

Correct an invalid output only through a separately preserved revision and
documented source rereading. Never edit the raw agent file to conceal the error.
If that becomes necessary, extend the importer with explicit revision lineage
before promoting the corrected artifact.

## Other artifacts

Store reports, label tables, audit results, configurations, experiment metrics,
predictions, and appropriate samples/checkpoints under `initiative/`. Keep large
external artifacts in `provenance/external-artifacts.json` with their immutable
locations, sizes, hashes, and status. Existing incomplete/legacy artifacts must
retain those labels. Do not archive credentials, signed URLs, or raw tool logs
that contain them.

To partition a new verified inventory while preserving rows:

```bash
python -m rsna.archive --refresh --inventory /absolute/path/dicom_inventory.parquet
```

The destination `initiative/source/inventory/` must be empty. The command checks
source metadata and exact table equality after reading the partitions back.
For the current dataset this is already done; do not recreate it on every run.

After adding/updating legitimate archive files:

```bash
python -m rsna.archive --refresh
python -m rsna.archive
ruff check .
ruff format --check .
mypy rsna
python -m pytest -q
git diff --check
```

To record a partial cache snapshot, use
`python -m rsna.archive --refresh --cache-progress /absolute/path/to/cache`.
This records writer-reported counts, bytes, warnings, and transfer syntaxes
with a timestamp. It explicitly does not certify full cache integrity.

Review the diff and stage explicit files/directories. CI verifies the committed
archive's membership, hashes, and review index. It allows accurately recorded
pending stages; release preparation separately requires
`python -m rsna.archive --require-complete` and the DICOM/training gates.
