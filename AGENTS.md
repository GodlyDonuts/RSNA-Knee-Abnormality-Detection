# RSNA Knee: current agent guide

The owner wants competitive model improvement and a fully current GitHub handoff.
Implement directly: **Devin is unavailable**. Do not create issues or wait for
Devin. Preserve healthy work while pursuing concrete, falsifiable experiments.
Software tests, parameter count, model downloads and training loss are not score
improvements; report actual measured results without promising first place.

## Read first

1. `docs/current-state.md`: timestamped scores, active runs and actual evidence.
2. `docs/research-brief.md`: mechanisms, completed comparisons and negative results.
3. `docs/research-agent-handoff.md`: exact assets, evidence map and open questions.
4. `docs/codex-continuation.md`: original jobs, live owners, roots and execution.
5. `initiative/research/owner-deadline-program-20261017-v1/README.md`: latest supplied
   paper, its limitations and the new S6 reproduction direction.

Update these guides **in place** after material changes. Original receipts and
fresh live state override dated narratives. The complete previous root guide is
preserved verbatim at `docs/AGENTS.history-20261017.md`; its old scores,
pending/running snapshots and next-action instructions are historical. Other
historical release/plan documents now link back to the current guides.

## Current scientific distinctions

- Best independently MRI-trained model: N3 CoAtNet E16, public AUC **0.894**.
- Best reproduced external checkpoint: nartaa CoAtNet, **0.949**. We reproduced
  inference, not its full training. Context/regional/full-field variants scored
  **0.948**; none establishes an improvement.
- Full-parent, sampled full-encoder and expanded N3 work is already owned and
  running. Inspect exact original owners before acting. A missing log/SSH link
  is observation loss, not terminal evidence or authority to restart a GPU.
- New S6 checkpoint/source/anatomy are privately verified, and the unchanged
  CPU stack loads strictly. Original CPU-only MRI gate139151928v1 completed;
  its1586windows and full CPU classifier reference942136 pass. Original H100
  probe942047 is pending; finite S6 T4 owner16380 waits capacity/funding without
  a GPU push. Never duplicate those originals. Its
  reported **0.950 is external**, not a project inference/score result. The
  unchanged notebook runs both S6 and CoAtNet; include both in runtime estimates.
- Released encoders exposed internal studies. Head-only cross-validation on
  their features is not independent whole-model OOF. Protected clinical values
  remain unused by us; weak-label agreement is not clinical truth.

## Execution, assets and resources

Inspect current Git, original Slurm/source/output bindings, epoch sidecars,
publication journals, canonical Kaggle records and fresh quotas before mutation.
Never duplicate a job, Dataset, notebook, upload or competition entry. Reconcile
uncertainty read-only; preserve failures and original identities. Keep healthy
numerical recipes, accepted masks/folds and exact source/weight/runtime bindings
unchanged. New mechanisms require an explicit matched comparison and actual
hardware/input/recovery evidence appropriate to that change.

Kaggle uses the single canonical ledger
`/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3` and coordinator. Refresh
actual account allowance before an entry; respect five daily submissions and
actual two-batch-GPU-session capacity. Use exact accepted offline versions and
`submission.csv`; never upload the public three-row example as a competition
result. Preserve original finite owners and their intents instead of starting
another actor for the same operation.

Keep at most two concurrent Newton GPU nodes and minimum CPUs. Use actual
funding records in the execution handoff, retaining 20 GiB byte and 5,000 shared
file reserves after liabilities. Do not delete originals for speculative space
or double-count paid bytes. Reuse exact original asset/whole-byte proofs instead
of repeating bulk transfers or hashing without a concrete concern. Privately
preserve each completed immutable epoch once with full download/size/SHA,
source, weight, licence, environment and cohort provenance. Use the canonical
shared publication lock; do not bypass it or retry an uncertain publication.

Credentials stay in ignored0600.env/RAM/stdin/ASKPASS, never argv/logs/Git.
Large original pixels, caches, environments, models and checkpoints stay on
approved private storage. Publish sanitized aggregate findings and exact pins,
not private clinical records, restricted weights or temporary authenticated URLs.

## Data and archive contract

Use the accepted release, frozen `protocols/report_weak_v2.md` and current
`initiative/reviews-v2/` evidence. Do not use legacy labels as the corrected
manifest. Keep all 12 targets in `rsna/schema.py` order. Silence is not negative;
uncertain/not-mentioned cells remain masked. Gold is evaluation-only. Report
extraction and DICOM decoding are not diagnostic rereading or proof of perfect
labels. Preserve original reports, judgments, source/provenance and corrections.

The historical v1 workflow is `wfr-b7eeba0fae7f4af18b2f5a869214114c`; corrected
review run is `codex-20260920-rubric-v2`. Resume only actually unfinished work,
never repeat completed reviews. Archive actual draft/reviewed JSONL with original
agent/session IDs and hashes; do not replace original agent output. The v1 archive
is intentionally incomplete; current v2 review coverage is checked separately.

## Git and verification

The owner explicitly authorized direct main integration and a monorepo without
requiring PRs. Commit verified milestones and use `git push origin HEAD:main`.
Preserve original research reports; label obsolete guidance as historical rather
than silently rewriting its claims. Keep small audit/source/provenance receipts
in Git and preserve verified milestones in the private HF dataset.

Before pushing, run `ruff check .`, `ruff format --check .`, `mypy rsna`,
`python -m pytest -q`, and `python -m rsna.archive`. Record existing failures
honestly; do not reformat archived original evidence to hide historical lint.
Validate current v2 coverage with:

```bash
python -m rsna.review_progress --inputs initiative/reviews/inputs \
  --root initiative/reviews-v2 --rubric protocols/report_weak_v2.md --require-complete
```

After adding verified archive members, refresh its file manifest and verify it;
do not modify original clinical artifacts or synthesize missing review stages.
Run meaningful focused checks for changed numerical code and retain actual
results. Check guide links, source pins and secret/large-artifact exclusions
before publication. Tests and checkpoint preservation establish operability,
not MRI accuracy.
