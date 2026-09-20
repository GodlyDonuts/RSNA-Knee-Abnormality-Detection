# RSNA initiative: start here

Read `initiative/README.md`, `docs/continuation.md`, and `docs/decisions.md` before
continuing work. This repository is the durable handoff record. Do not rely on
the originating Devin VM or chat history.

- Inspect `initiative/reviews/index.json` for actual archived coverage.
- Run `python -m rsna.archive` to verify file hashes, exact batch membership,
  source report text, evidence quotations, and the review index.
- Check `initiative/audits/` for measured results. A partial snapshot is not a
  completed DICOM audit or a corrected data release.
- Continue workflow `wfr-b7eeba0fae7f4af18b2f5a869214114c` using the recovery
  instructions. Do not start another full review from scratch by accident.
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
- Commit and push meaningful verified progress incrementally to
  `devin/1789934150-training-readiness` while PR #1 is open. Do not assume `main`
  already includes this work; do not merge without authorization.
- Run `ruff check .`, `ruff format --check .`, `mypy rsna`,
  `python -m pytest -q`, and `python -m rsna.archive` before pushing.
- Update the handoff archive and PR description when milestones change. Store
  small final manifests, audits, predictions, and experiment metadata here even
  when also publishing them to Hugging Face.
