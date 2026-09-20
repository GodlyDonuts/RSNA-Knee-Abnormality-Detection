# Corrective data preparation, 2026-09-20

The user authorized direct main integration on 2026-09-20 and requested one monorepo without a PR workflow. The corrective pipeline commit is integrated into main at `423733e`; both prior PRs are integrated. The active worktree uses local branch `codex/data-readiness-audit`, tracking `origin/main`. Push verified milestones with `git push origin HEAD:main`.
The legacy archive remains unchanged. Its rubric accepted low-severity findings that do not satisfy the organizer criteria. Continue the new versioned review run, `codex-20260920-rubric-v2`, under `protocols/report_weak_v2.md`; do not resume the old rubric as a corrected-label workflow.

## Current gates and evidence

- All five fresh authenticated Kaggle metadata downloads match `initiative/source/` byte for byte. `rsna.preflight` independently checked source/inventory relationships and recorded 54 duplicate-report groups covering 204 studies.
- `initiative/reviews-v2/` stores completed drafts/reviews, immutable per-artifact identity/hash sidecars, review attention flags, and the compiled progress index. Original blinded inputs remain in `initiative/reviews/inputs/`.
- Batches 000, 001, 002 and 088 now have both validated passes: 157 reports. Preserve their prereads and adjudication history. Batches 003..005 are being read next; use the index for subsequent progress. Stage unfinished work outside `initiative/` to keep archived milestones stable.
- Every report requires a reader plus a separate source-reading reviewer. Reports alone create weak labels, with ambiguous findings masked. Gold values must not be inspected to tune extraction. All official evaluation studies remain excluded from training.
- Cache version is `series-u16-v3`; old v2 metadata/hashes must not be relabeled v3. The first 100 series / 3,182 slices passed the new decoder and cache audit. This pilot is not the full audit.
- The larger isolated pilot covers 150 training studies plus all 3 public test examples, 848 series and 28,882 slices. Its independent audit found no cache warnings or cross-study pixel duplicates within that subset. See `initiative/audits/pilot-realdata-v3/`.
- The real-data pipeline diagnostic passed 126 optimizer updates over two complete epochs; continuous and resumed loaded checkpoints match exactly, including model, optimizer and RNG state. All three public test predictions match template order and contain 12 finite probabilities. `initiative/audits/pilot-realdata-smoke-v3/` archives the tiny checkpoints, commands, manifest, scope, predictions and hashes. These are diagnostic weights, not competition training initialization. The full decoded-pixel graph and final-release smoke are still required.
- A full streaming image run is in progress in the calling workspace's `work/cache256-v3/`. Its supervisor, status and log are `work/full_cache_job.py`, `work/full_cache_job.json`, `work/full_cache_job.log`. Inspect the active process before restarting. A successful preprocess is followed automatically by the full `rsna.audit` run into `work/dicom-audit-v3/`. Use 2 series workers on the 8 GiB host and preserve 20 GiB free disk.
- The one-header-per-study audit covered 4,410 released studies and found 4,410 distinct nonempty 12-character PatientIDs, all lacking issuer metadata. No repeated identifier groups were found. This does not establish true patient identity or consistency across every header. `initiative/audits/patient-header-audit/` preserves hashed identifiers and source-byte provenance. The exact run script is archived in `initiative/provenance/patient_header_scan.original.py`; it expects a workspace `work/` containing the repository as `recovered/` and uses environment credentials only.
- Assistant visual checks cover six native/cache slice pairs and 48 spaced cached slices, across all 3 planes. No gross crop/blank/encoding artifact was observed in those samples. An edge slice is very dark but later anatomy is useful, so it is retained. Downsampling softens detail; full-resolution source remains authoritative. See `initiative/audits/visual-*.json`.

## Compile and validate new review outputs

Per-artifact sidecars go to `initiative/reviews-v2/provenance/batch-NNN-{draft|reviewed}.json`. Use the actual agent identity, frozen rubric hash, canonical source-report fingerprint, input-file hash, exact UID list, output hash, study count, stage and notes. Review sidecars also bind the exact parent draft SHA256. Never generate missing annotations programmatically; code may assemble and validate individually reasoned decisions.

```bash
python -m rsna.review_progress --inputs initiative/reviews/inputs --root initiative/reviews-v2 --rubric protocols/report_weak_v2.md
# The following must fail until all 89 batches have both valid passes:
python -m rsna.review_progress --inputs initiative/reviews/inputs --root initiative/reviews-v2 --rubric protocols/report_weak_v2.md --require-complete
```

The compiler checks evidence, identity, source/rubric/parent hashes and coverage. These checks cannot replace semantic review. It never certifies training readiness. Incomplete in-progress artifact/sidecar pairs are not eligible for release.

After the full image and review gates pass, use the updated `rsna.labels` invocation in README. Then perform real-data optimizer, exact resume, gold-inference and public-example submission checks. Preserve masked unknowns and investigate any class collapse; do not invent positives to balance classes.

## Preserve verified progress

Freeze completed batch outputs before updating their archive snapshot. Run `python -m rsna.archive --refresh` then `python -m rsna.archive`, all tests, Ruff and mypy before pushing. The old archive's `--require-complete` concerns the historical v1 run, so use `rsna.review_progress --require-complete` for the v2 run.

The supplied HF mirror is currently public. Do not publish new competition reports, label artifacts or images to an unrestricted destination; follow the verified competition data-sharing rules. Keep deliverable preparation local and use the existing private GitHub repository for the authorized handoff. Public code can be separated from restricted data if a later release requires it.
