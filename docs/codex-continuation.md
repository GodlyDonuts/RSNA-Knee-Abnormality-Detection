# Corrective data preparation, 2026-09-20

The user authorized direct main integration on 2026-09-20 and requested one monorepo without a PR workflow. The corrective pipeline commit is integrated into main at `423733e`; both prior PRs are integrated. The active worktree uses local branch `codex/data-readiness-audit`, tracking `origin/main`. Push verified milestones with `git push origin HEAD:main`.
The legacy archive remains unchanged. Its rubric accepted low-severity findings that do not satisfy the organizer criteria. Continue the new versioned review run, `codex-20260920-rubric-v2`, under `protocols/report_weak_v2.md`; do not resume the old rubric as a corrected-label workflow.

## Current gates and evidence

The full-image milestone (2026-09-21) includes 54 completed v2 review batches covering 2,657 reports and 31,884 target decisions. The compiled index remains authoritative as subsequent batches arrive. All preread hashes and exact quotations were validated. Full review coverage and global semantic adjudication remain required.

The complete endpoint survey covers 48,772 headers, all 24,386 series and 4,410 studies without request failures. Nine independently examined plane candidates preserve official routing; six accepted overrides in study045:14 have a separately bound effective series table. Four studies contain two visually distinct knee protocols; their aggregate report supervision remains masked. These technical image checks are not diagnostic MRI rereads.

The corrected full cache has passed `rsna.audit`: 819,635 slices, 24,386 series, 4,410 studies, zero cache warnings, one cross-study nonconstant pixel edge and no released train/test overlap. Exact-series selection retains all 4,407 training studies and 24,338 of 24,371 series; 33 fully equivalent model slots are excluded. All 15 released test series remain. The exact 077:22 pair shares one slot, while the similar 042:40 pair remains separate. Every input cache file was checked again before export. See `initiative/audits/full-image-recovery-v1/` and `docs/series-equivalence.md`.

All 200 tests, Ruff and mypy pass. Root independently reviewed the recovery and inference changes, ran 26 recovery tests and 35 inference/pipeline tests, and verified every reused metadata hash and revision-only semantic delta. Held-out inference now binds the exact training series table, study manifest and cache source revision; separate test inputs record their own provenance. See `docs/source-recovery.md` and `docs/inference-bindings.md`.

Reports042:14 and042:41 explicitly combine both knees. Independent correspondence adjudication accepted 16 additional eligibility masks; the immutable original extractions and both readings remain preserved. These are report-to-image correspondence masks, not claims that the imaged studies contain both knees. The registry now contains 27 accepted target changes, while mandatory global cartilage and bone terminology QA remain pending. No original clinical artifact is overwritten.

GitHub and Hugging Face are private checkpoint destinations. HF source revision `0b5e410288232f570c69ba4f44723356e6ae6ab7` replaces exactly two invalid mirror encodings with verified official Kaggle files; the original pinned `52f995f4568e36bfa5fbbaf69a89af41226b4f5d` remains preserved. A verified private preparation snapshot of GitHub `31e4ff8d456e37fffc16fcd09f37ca613d2d7c02` was published with the source correction. Subsequent receipts establish newer snapshots. No finished training release has been published; checkpoints remain explicitly incomplete until all gates pass.

- All five fresh authenticated Kaggle metadata downloads match `initiative/source/` byte for byte. `rsna.preflight` independently checked source/inventory relationships and recorded 54 duplicate-report groups covering 204 studies.
- `initiative/reviews-v2/` stores completed drafts/reviews, immutable per-artifact identity/hash sidecars, review attention flags, and the compiled progress index. Original blinded inputs remain in `initiative/reviews/inputs/`.
- Use the index to select unfinished batches; preserve every completed preread and adjudication history. Stage unfinished work outside `initiative/` to keep archived milestones stable.
- Every report requires a reader plus a separate source-reading reviewer. Reports alone create weak labels, with ambiguous findings masked. Gold values must not be inspected to tune extraction. All official evaluation studies remain excluded from training.
- Cache version is `series-u16-v3`; old v2 metadata/hashes must not be relabeled v3. The first 100 series / 3,182 slices passed the new decoder and cache audit. This pilot is not the full audit.
- The larger isolated pilot covers 150 training studies plus all 3 public test examples, 848 series and 28,882 slices. Its independent audit found no cache warnings or cross-study pixel duplicates within that subset. See `initiative/audits/pilot-realdata-v3/`.
- The real-data pipeline diagnostic passed 126 optimizer updates over two complete epochs; continuous and resumed loaded checkpoints match exactly, including model, optimizer and RNG state. All three public test predictions match template order and contain 12 finite probabilities. `initiative/audits/pilot-realdata-smoke-v3/` archives the tiny checkpoints, commands, manifest, scope, predictions and hashes. These are diagnostic weights, not competition training initialization. The completed full pixel graph must feed final transitive leakage isolation; final-release smoke checks remain required.
- The baseline streaming job stopped with two invalid DICOM encodings, one in each of two series. Exact official source restoration and a separate recovery completed all 74 affected slices. Do not restart the old supervisor against the invalid baseline inventory. The effective cache is `work/cache256-v3-official-recovery-v1/cache`, with its exact corrected inventory beside it; the full audit is `work/dicom-audit-v3-official-recovery-v1/`. Reused NPZ hardlinks are immutable: never rewrite either old or new cache files in place. Preserve 20 GiB free disk. Source restoration, per-series lineage, independent visual limits, full audit and training/test selection artifacts are archived under `initiative/audits/source-pixel-representation-v1/` and `initiative/audits/full-image-recovery-v1/`.
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

The HF source mirror was freshly verified private on2026-09-21. Preserve restricted access for competition reports, labels and images. Validated preparation checkpoints can be mirrored privately with explicit incomplete status; the pinned original source revision stays unchanged. The private GitHub repository remains the code/review handoff. No public cleaned release is authorized by this workflow.
