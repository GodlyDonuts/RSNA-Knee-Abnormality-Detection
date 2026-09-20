# Continue from a new session

This file preserves the originating session's recovery instructions. PR #1 has
since merged, and the corrected rubric/cache run is documented in
[Codex continuation](codex-continuation.md). Use that guide for current work;
recover the old workflow only when collecting historical evidence.

## 1. Get the correct code and evidence

The working branch is `devin/1789934150-training-readiness`, in
[PR #1](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/pull/1).
If it has not merged, check out that branch rather than assuming the default
branch has the archive.

```bash
git fetch origin
git switch devin/1789934150-training-readiness
python -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.0.1
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m rsna.archive
ruff check .
ruff format --check .
mypy rsna
python -m pytest -q
```

No pre-commit configuration is present. The verified environment setup uses
Python 3.10 and CPU PyTorch. A repository blueprint was proposed/saved during
the originating session; these commands are the reproducible setup even if a
new machine does not inherit it. GPU training requires the appropriate CUDA
PyTorch build and separate hardware verification.

Read `initiative/reviews/index.json`, `initiative/audits/`, and
`docs/decisions.md`. The archive's missing stages are intentional indicators of
unfinished work, not permissions to synthesize missing labels.

## 2. Recover the review work

The recovery instructions below preserve the original workflow and its evidence.
Its rubric has since been found inconsistent with the competition's severity
thresholds; recovered outputs do not qualify as corrected competition labels.
New report passes must follow `protocols/report_weak_v2.md` and be stored under
`initiative/reviews-v2/`. Keep the original files immutable. Each new draft and
review provenance record must bind `rubric_sha256` to the frozen rubric and
`source_reports_sha256` to `rsna.labels.source_reports_fingerprint` of all original
blinded reports. Reader and reviewer must have different agent and session IDs.

- Parent session: https://app.devin.ai/sessions/96aa9aa1ea7d4b3d9dd152915b9d4efc
- Workflow: `wfr-b7eeba0fae7f4af18b2f5a869214114c`
- Executable portable workflow: `workflows/report_review.py`
- Exact original script: `initiative/provenance/report_review_workflow.original.py`
- Every archived child session/agent ID: `initiative/reviews/provenance.json`

Use Devin's `get_workflow_output` for the run. Discover the current
`devin_session_interact` schema with `devin_mcp list_tools` before inspecting child
sessions. Its `get` action returns the structured output (`artifact_url`,
`n_studies`, `notes`); `get_attachments` can recover a file if necessary.
Download attachments with `download_attachment`, then import actual files using
the [archive procedure](archive-maintenance.md). No attachment URL is necessary
to read the outputs already in GitHub.

If the run is interrupted, invoke the `dynamic-workflows` skill and resume with
the same run ID and unchanged prompt/schema content. The original script used
`/home/ubuntu/rsna-hf-repo/train.csv` and wrote
`/home/ubuntu/rsna-readiness/report_review_results.json`. The portable copy uses
the same names under `Path.home()`. Copy the tracked `initiative/source/train.csv`
to that source location first if the HF clone is absent; it is the original
file. Changing source/output filesystem locations does not justify changing
reports, rubric, ordering, or schema. Do not launch another full 178-agent run
without first checking existing results and replay behavior.

Six agents run concurrently on separate VMs. Batches contain 50 reports, except
the final batch of seven. All input UIDs are sorted lexically. Gold columns are
excluded from child prompts. A separate agent re-reads every report and checks
the draft. Completed agents free slots automatically; queued annotations can
precede queued reviews. An annotation is not an independent review.

If a new session cannot operate a run owned by the original session, inspect its
recorded child outputs and collect them before arranging continuation. Do not
discard the archived work. Keep original agent files immutable. Any correction
requires a separately preserved artifact, exact source evidence, and provenance.

## 3. Resume image preparation

The source inventory and all required metadata are already in GitHub. The
original session's working cache is:

```text
/home/ubuntu/rsna-readiness/cache256-v2
```

That is an optional same-VM resume location, not a dependency available in another
session. It contains the historical v2 transform; current `series-u16-v3` preparation
must use a new output directory and rebuild those series. Its process was started
in shell `d4b057`; shell IDs are session-local.
Check whether it is still running before starting another writer. Inspect
`progress.jsonl` and disk space, then reuse verified cache pairs. Do not point
two cache processes at the same output directory.

On a fresh machine, use a persistent directory with enough space:

```bash
mkdir -p "$HOME/rsna-readiness"
python -m rsna.preprocess \
  --manifest initiative/source/inventory \
  --output "$HOME/rsna-readiness/cache256-v3" \
  --source-revision 52f995f4568e36bfa5fbbaf69a89af41226b4f5d
python -m rsna.audit \
  --inventory initiative/source/inventory \
  --cache "$HOME/rsna-readiness/cache256-v3" \
  --revision 52f995f4568e36bfa5fbbaf69a89af41226b4f5d \
  --output "$HOME/rsna-readiness/dicom-audit"
```

Public HF reads work without a credential but can be rate-limited. Inject an
existing `HF_TOKEN` from secret storage for authenticated reads. Do not put the
token in a command, Git file, or URL. The current cache implementation uses
authenticated LFS requests when available and observes rate-limit reset headers.
Kaggle download previously returned HTTP 401; the HF mirror avoids that access
dependency. The mirror does not contain hidden competition-test studies.

All 24,386 series / 819,635 slices must pass the full audit. Source path coverage
alone is not successful decoding. Inspect warnings and duplicate-pixel groups.
Train/test identical-pixel pairs block release pending investigation. Cache
generation stops below 3 GiB free; do not suppress this guard.

To independently regenerate the original inventory, use the pinned HF Git
revision and the archive index:

```bash
gzip -dc initiative/source/zip_files.txt.gz > "$HOME/rsna-readiness/zip_files.txt"
GIT_LFS_SKIP_SMUDGE=1 git clone \
  https://huggingface.co/datasets/GodlyDonuts/RSNA-Data "$HOME/rsna-hf-repo"
python -m rsna.inventory --repo "$HOME/rsna-hf-repo" \
  --revision 52f995f4568e36bfa5fbbaf69a89af41226b4f5d \
  --metadata-root initiative/source \
  --original-paths "$HOME/rsna-readiness/zip_files.txt" \
  --output "$HOME/rsna-readiness/dicom_inventory.parquet"
```

Skip the clone command when that checkout already exists; fetch the pinned
revision instead.

## 4. Finish labels and real-data verification

```bash
# Verify historical evidence; this is not corrected-label approval:
python -m rsna.archive
python -m rsna.labels \
  --train initiative/source/train.csv \
  --drafts initiative/reviews-v2/drafts --reviews initiative/reviews-v2/reviewed \
  --review-provenance initiative/reviews-v2/provenance.json \
  --rubric protocols/report_weak_v2.md \
  --inventory initiative/source/inventory \
  --source-revision 52f995f4568e36bfa5fbbaf69a89af41226b4f5d \
  --dicom-audit "$HOME/rsna-readiness/dicom-audit/dicom_audit.json" \
  --output "$HOME/rsna-readiness/release"
```

The label command fails until the new two-pass review is complete and bound to the
source/rubric hashes and the complete current-version DICOM audit matches its source
inventory and series index. Do not manufacture provenance or repoint old outputs to
the new rubric.

Then check target distributions, observed coverage, blinded gold agreement,
reviewer changes, report/pixel duplicates, quarantine, and split disjointness.
Do not inspect gold to tune the annotation rubric and then call the result an
unbiased evaluation.

Follow the README's training/resume/gold-inference/sample-submission commands
using the corrected manifest. Run a short real-data optimizer smoke test first,
then exact epoch-boundary resume with unchanged settings. Record measured
metrics, configurations, artifact hashes, and throughput. A CPU smoke test does
not establish competitive accuracy or the nine-hour Kaggle GPU runtime.

## 5. Preserve progress

Commit newly completed annotation/review outputs, final small manifests, audit
JSON/Parquet, corrected label tables, prediction CSVs, smoke metrics, suitable
checkpoints, and experiment configurations in GitHub. Mirror releases in HF
only after their acceptance gates pass. Include per-series cache metadata and
hashes with the final audit, even when the large cache volumes remain external.

Refresh `initiative/manifest.json` with `python -m rsna.archive --refresh`,
verify with `python -m rsna.archive`, run the required checks, and push focused
commits. Update `initiative/audits/`, the continuation guide when facts change,
and PR #1's description. The full corrected release must include reciprocal
immutable GitHub/HF references, as described in `docs/releases.md`.
