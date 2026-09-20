# RSNA initiative: start here

Read `initiative/README.md`, `docs/continuation.md`, and `docs/decisions.md` before
continuing work. This repository is the durable handoff record. Do not rely on
the originating Devin VM or chat history.

The current corrective run is described in `docs/codex-continuation.md`.
Use the frozen `protocols/report_weak_v2.md` and `initiative/reviews-v2/`.
The earlier report rubric conflicts with the organizer's severity definitions;
its preserved outputs are historical evidence, not current training labels.

- Inspect `initiative/reviews-v2/index.json` for current archived coverage.
- Run `python -m rsna.archive` to verify file hashes, exact batch membership,
  source report text, evidence quotations, and the review index.
- Check `initiative/audits/` for measured results. A partial snapshot is not a
  completed DICOM audit or a corrected data release.
- Preserve workflow `wfr-b7eeba0fae7f4af18b2f5a869214114c` and its outputs as the
  historical v1 run. The deliberate corrected run is `codex-20260920-rubric-v2`.
  Resume only unfinished v2 batches; do not repeat completed passes by accident.
- Archive every newly completed draft and reviewed JSONL with its session/agent
  IDs, completion notes, and hash. Never replace the original agent output.
- Keep all 12 target columns in `rsna/schema.py` order. Silence is not negative.
  Uncertain and not-mentioned findings stay masked. Gold is evaluation-only.
- Never claim that report extraction or technical DICOM decoding is a diagnostic
  rereading of the MRI images. Earlier automated-label readiness claims were
  retracted; do not repeat them.
- Source reports, review data, inventories, experiment records, and suitable
  DICOM/cache/checkpoint artifacts belong in GitHub. Large collections also need
  pinned external locations, hashes, and rebuild commands. Keep credentials and
  authenticated temporary URLs in secret storage.
- Do not load `initiative/legacy/` labels as the corrected training manifest.
  That directory preserves historical work, including unvalidated reference code.
- The user explicitly authorized direct main integration on 2026-09-20 and
  requested a monorepo without a PR workflow. Both prior PRs are integrated.
  Commit verified milestones and push to `origin/main`. The active worktree's
  local branch tracks main; use `git push origin HEAD:main` explicitly.
- Run `ruff check .`, `ruff format --check .`, `mypy rsna`,
  `python -m pytest -q`, and `python -m rsna.archive` before pushing.
- Validate v2 coverage with `python -m rsna.review_progress --inputs
  initiative/reviews/inputs --root initiative/reviews-v2 --rubric
  protocols/report_weak_v2.md`. The historical archive completeness flag applies
  to v1; use this command's `--require-complete` for the current run.
- Update the handoff archive and continuation guide when milestones change. Store
  small final manifests, audits, predictions, and experiment metadata here even
  when also publishing them to Hugging Face.
