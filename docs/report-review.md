# Report review

The historical workflow below used an obsolete rubric that included trace effusion
and did not enforce the competition's ligament severity thresholds. Preserve its
outputs for provenance. Corrected competition supervision uses the frozen
[`report_weak_v2.md`](../protocols/report_weak_v2.md) and new artifacts under
`initiative/reviews-v2/`, never overwritten historical files. Both passes need
distinct agent/session identities, artifact hashes, the frozen `rubric_sha256`, and
`source_reports_sha256` from `rsna.labels.source_reports_fingerprint` over the full
blinded source report dictionary. These are mandatory label-release checks.

`workflows/report_review.py` is the executable workflow used for the replacement
labels. Run it through Devin's `run_workflow`, not plain Python: `agent`, `log`, and
`register_workflow` are injected by the workflow runtime.

1. Make the pinned original `train.csv` available locally; point `SOURCE` at it.
   Set `OUTPUT` to a persistent location. The workflow passes only the UID and
   report to agents, never gold target columns.
2. Invoke the `dynamic-workflows` skill, then `run_workflow` with the script's
   absolute path and a descriptive workflow name.
3. The workflow sorts all 4,407 reports by study UID, creates 89 batches of at most
   50, and keeps up to six separate-VM agents active. Each batch gets a reader and
   a separate reviewer. Completed agents automatically free a slot for queued
   work. Each agent consumes ACUs. A reviewer must re-read every report and return
   a full corrected annotation, not merely validate formatting.
4. Save the run ID. If interrupted, resume with that ID and identical prompts.
   Completed calls replay; failed calls run again. A failed batch blocks release.
5. Download every `artifact_url` in the result using Devin's
   `download_attachment`; the attachment URLs require authenticated storage
   access, not anonymous HTTP requests. Preserve the `batch-NNN-draft.jsonl` and
   `batch-NNN-reviewed.jsonl` names in separate folders.
6. Run `python -m rsna.labels` after the image audit. It enforces complete,
   one-to-one coverage, exact 12-entry arrays, allowed states, and source-substring
   evidence. Missing or invalid records must be corrected by rereading the source,
   never filled with automatic negatives.

Archive every completed output in GitHub immediately using
[the archive procedure](archive-maintenance.md), even while other batches are
running. The source reports and all blinded batch inputs are already under
`initiative/`. Preserve original bytes, completion notes, and session/agent IDs.
The generated review index makes incomplete coverage visible to future sessions.

The complete target rubric is embedded in the workflow to keep agent prompts
self-contained and replayable. Key distinctions: ligament identity, tear versus
meniscal degeneration, compartment-specific OA versus isolated chondromalacia,
effusion versus synovitis, and contusion versus nonspecific marrow edema.
History, indication, and postoperative changes must not be mistaken for a definite
current finding.

Original-language evidence is retained. Every non-`not_mentioned` state needs a
verbatim quote; `not_mentioned` needs empty evidence. Store ambiguity in `notes`.
Unknown and uncertain findings are masked during learning.

Compare the final review with official gold only after the blinded annotation
stage. The audit reports coverage, confusion counts on observed targets, and how
often reviewers changed the initial state. These are report-extraction results;
they do not establish radiologist-equivalent image accuracy.

Changing the rubric after inspecting gold changes the evaluation protocol. Record
that explicitly and obtain a new independent evaluation set before presenting an
unbiased diagnostic performance claim.
