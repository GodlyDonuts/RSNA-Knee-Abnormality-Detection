# Current annotation task procedure

An assignment names one canonical batch and either `draft` or `reviewed` stage.
Read `protocols/report_weak_v2.md` first. All target decisions must be personally
reasoned from the original report; scripts may only assemble explicit decisions
and validate source quotations, schema and coverage. Do not substitute regex,
NLI, public labels, or a default-filling procedure. Do not open official target
values or legacy annotations.

Read the full `initiative/reviews/inputs/batch-NNN.json`, conveniently in groups
of ten reports. Preserve original language, laterality, finding versus history,
negation, severity and uncertainty. Reason through all twelve targets for every
report. Every nonmissing state requires a contiguous original-language source
quotation; `not_mentioned` requires empty evidence. Keep ambiguous targets masked.

For a reader task, write
`initiative/reviews-v2/drafts/batch-NNN-draft.jsonl`. For a reviewer task, first
persist a complete independent source reading under
`initiative/reviews-v2/evidence/batch-NNN-preread.jsonl`, then open the v2 draft and
reader flags. Compare every target and reread source as needed. Write a complete
corrected `initiative/reviews-v2/reviewed/batch-NNN-reviewed.jsonl`, not a patch or
schema-only approval. Preserve the preread and original draft. Record unresolved
ambiguities and every changed state in the reviewer sidecar.

Stage all unfinished files outside `initiative/`, for example in the calling
workspace's `work/annotation-staging/batch-NNN/`. Publish completed artifact and
sidecar pairs only after validation, so archival snapshots remain stable while
other batches are being read. Do not discard intermediate reasoning.

Final annotation fields
are exactly `study_uid`, `states`, `evidence`, `notes`; both arrays have twelve
entries in schema order. Call `rsna.labels.read_reviews` on the specific batch
suffix to validate the completed artifact against its blinded inputs.

Write the sidecar at
`initiative/reviews-v2/provenance/batch-NNN-{draft|reviewed}.json`. Include:

- `stage`, `rubric_version: report-weak-v2.0`, actual canonical agent identity in
  `agent_id` and `session_id`, and `identity_kind: codex_agent_task`. These are
  Codex task identities, not invented external provider session identifiers.
- `workflow_run_id: codex-20260920-rubric-v2`.
- `rubric_sha256: 138c5cc33b92901d57dbf5fbe7e3a9878b73d25b0af6934908d0052cd6786014`.
- `source_reports_sha256: 5c18838a09e9eec6522966555694fca83a9bae67e4d7c5c68592207ee4b60ed3`.
- `source_sha256`: hash of the exact blinded batch JSON file; `study_uids`: its
  exact sorted UID list; `n_studies`: actual count.
- `sha256` or `output_sha256`: hash of the completed JSONL; repository-relative
  source/output paths and meaningful completion `notes`.
- For reviewers, `parent_artifact_sha256` equal to the exact draft hash and the
  independent preread file path/hash. The reviewer must be a different agent
  from the draft's reader. Never review your own draft.

Do not change the frozen rubric, another agent's output, code, or the archive
manifest during annotation work. Flag a suspected protocol defect to the parent
instead of silently changing definitions. The parent compiles progress and
archives completed batches. Coverage verification is not semantic validation.
