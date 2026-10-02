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

## Owner's research directive, October 1

The owner explicitly asks for independent, creative research that pushes beyond
repeating established recipes. Make this a standing project priority.

- Seek useful mechanisms from anatomy, physics, natural systems and other fields;
  translate each analogy into a specific model or data hypothesis. Do not claim
  novelty without checking prior work, or assume novelty improves performance.
- Do not dismiss an unconventional or apparently contradictory idea solely
  because it conflicts with the usual recipe. State the competing mechanisms
  and design an experiment that can distinguish them.
- For each proposed experiment, record the failure it addresses, the predicted
  effect, a matched control, the compute cost and evidence that would reject it.
  Prefer experiments that distinguish explanations over undirected scaling.
- Improve how the solo DINOv3 model observes an examination: information coverage,
  spatial relationships, acquisition diversity and label-compatible external
  supervision are research directions, not established improvements.
- Pursue promising work autonomously within existing authorization. Preserve the
  accepted data boundaries, exact provenance, source rights and resource limits.
  Do not change live experiments merely to fit a new idea.
- Promote ideas through measured held-out performance and reliable MRI-only
  inference. Preserve negative results. Software tests, more data and confident
  prose do not demonstrate clinical accuracy or a winning Kaggle score.

See `docs/creative-research-plan.md` for the current hypothesis sequence and the
conditional external-data investigation.

## October1 private source initialization, current observation loss

Stanford images are acquired/verified and bounded actual adapter pilot passed.
Full-engine recovery867512 is lastPENDING, full1129/119 fitting unsubmitted. New
optional canonical RSNA initialization accepts only its completed full source
checkpoint/terminal SHA and transfers encoder adapters only. Fourteen synthetic
CPU tests passed actual RSNA full/stop1/resume equality; full QA1638/Ruff258/mypy43.
No new GPU/clinical pass. Require private owner-only node-local outputs; source-
derived competition weights remain unadmitted. Read mrnet-rsna-private-
initialization-v1 and current continuation. Cisco reports Disconnected; Newton
DNS unavailable. Never restart last-seen healthy native867367/index867494 or
duplicate867512 because observation is lost. New milestones await unreachable
shared lock for HF mirroring. Dated snapshots below preserve history.

## October1 native training and Stanford pixel-only milestone

Native867367 RUNNING on evc31,47 finite real updates at20:46UTC, actual header/
CUDA/native384 admission passed. New full Stanford engine867478 still pending;
source fitting1129/119/10epochs not dispatched. Older source pilot867419 passed
actual four-update recovery only. Separate correctedCPU867493 completed all3750
Stanford volumes/1250image exams, raw private labels untouched; RSNA CPU867494
still running eligible native series. No overlap/clinical accuracy conclusion.
Read stanford-rsna-pixel-overlap-screen-v1 and native gpu-initial-progress.
Original failed CPU867491/867492 missing schema import preserved, v2isolated
closure fixed. Never duplicate jobs or publish private source signature records,
raw pixels or learned Stanford weights. HF417905f3d6603d97a8e62ed767d4b693e8241fca
fully verified50 small code/metadata objects frommain6edefe1, no raw/labels/weights;
never repeat. Current continuation and actual receipts supersede old snapshots.

## October2 restored access and actual original recovery

VPN/SSH restored. Native867367 healthy, epochs1–3 complete/1052finite updates,
epoch4 training. Stanford same-engine867512 actual complete recovery passed;
full source fitting867694 dispatchedONCE, lastPENDING. Pixel867494 and private
matcher867696/867697 completed;92coarse pairs require original-image review,
not duplicate/leakage confirmation. Current guide/original audits supersede
prior access-loss snapshots. Preserve native epochs once/pinned coordinator,
leave healthy original jobs unchanged. No downstream transfer or new score.

## October2 actual private source fitting and native epoch4

Stanford867694 RUNNING on evc32,388finite updates/epoch3, first2 source epochs
privately byte/sidecar verified. Native867367 RUNNING,1328updates/epoch5; new
epoch4 privately full-download/hash verified. Do not repeat either job/epochs.
No source/native accuracy or transfer result yet. Candidate original-image
review protocol prepared only, no accepted wrapper/dispatch. Current guide
and original progress audits supersede pending/source-unstarted snapshots.

## October2 actual source-final and native six-epoch preservation

Stanford original867694 COMPLETED0:0, tenepochs/1420finite updates and allten
private source checkpoints verified. Independent119-exam source AUC replay:
ACL0.8621082621, genericmeniscus0.7215843858, abnormality0.9285087719.
These are source-task results, not RSNA/Kaggle or transfer improvement.
Native867367 RUNNING,1885updates/epoch7; epochs1–6 privately full-download/
hash verified once. No restart/source change. Read mrnet-full-source-final-v1,
native384-epochs-v1 and current continuation. Next accept bounded private
original-image review before downstream GPU transfer; original92coarse pairs
remain advisory. Source-derived weights remain private owning-node only.

## October2 native384 terminal and final comparison

Original867367 completed0:0/2810updates; allten native epochs privately fully
verified once. Read native384-fixed-final-v1 and latest continuation: internal
0.8630558226 vs strictuniform0.8606037925 is a small unestablished gain, paired
95% includes zero. No native Kaggle score. Preparing distinct final comparison
under exact C9 numerical provenance and fresh funding; original failed CPU
preparation868031 had zero cloud/ledger mutation. Distinct corrected rank32
preparation868037 passed; Dataset12322320v1 fully47-object verified and exact
private offlineT4 native epoch10 notebookv1 pushed ONCE, publicRUNNING.
The finite finish coordinator handles exact acceptance then one code submission;
inspect its intents/results and never duplicate it.
Never repeat completed sampling submissions or native publications.
