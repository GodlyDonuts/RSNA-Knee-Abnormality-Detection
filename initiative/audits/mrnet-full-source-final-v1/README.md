# Original Stanford full-source fitting: fixed-final result

Original job 867694 completed 0:0 on evc32 in 1:48:26. The unchanged F12
engine trained 1,129 source exams for ten epochs and 1,420 contiguous finite
optimizer updates, then evaluated its 119 source validation exams once.
Validation never entered fitting. Source ACL, generic meniscus and abnormality
labels remain separate from the twelve RSNA targets. No epoch selection used
these validation results.

| Source task | Fixed-final held-out AUC | Positive / negative exams |
| --- | --- | --- |
| ACL | 0.8621082621082621 | 54 / 65 |
| Generic meniscus | 0.7215843857634902 | 52 / 67 |
| Abnormality | 0.9285087719298246 | 95 / 24 |

Owning-node zero-GPU CPU job 867925 independently replayed pairwise AUC from
original private label CSVs and final predictions, matching the original
terminal within 1e-12. These are source-task AUCs on a small source holdout;
they are not RSNA or Kaggle scores and do not establish a transfer improvement.
All ten immutable source snapshots are privately retained and size/SHA/sidecar/
recipe/source-input/training-membership verified on evc32. Earlier whole-byte
proofs for epochs one and two were reused; only new epochs three through ten
were hashed. Retained bytes are 1,250,968,446, below the original 3 GiB cap.
Source pixels, label values, case membership, predictions and weights were
not collected to Mac, GitHub or Hugging Face.

Private output: `/tmp/rsna-mrnet-full-engine-retained-70ncrj4l/full` on evc32.
Science Git: `f12a319aa9d36d2fecbac4f2df445c3c1d731146`.
Final epoch-010/last checkpoint SHA:
`81093bb3e6450886d2d20c37804676af820664bf419716099894ae4cd5604f15`.
Worker terminal SHA:
`abc179c1540c5a620b3e60097c9697c6f267b28543518fdaf802477539d784c2`.
Outer terminal SHA:
`60037f114eb36cba5340d8730bda84bc26549f396253e109a87603c10f69764f`.
The two terminals decode identically; original serializers differ by a final
newline. Observer 867922 incorrectly required their raw-byte equality and
failed before any model/data mutation. Diagnostic 867924 and corrected
read-only observer 867925 verified the original receipts. This observer issue
is not a training failure; the original fit was never restarted.

Read `owning-node-final-verified.json`, `finite-training-terminal-proof.json`,
`terminal.json`, `run-start.json`, `source.json` and `final-slurm.txt` for actual
bindings and aggregates. Source checkpoint/publication rights remain private
research only. Before any RSNA transfer require original-image review of the
92 advisory cross-source candidate pairs, separately funded actual GPU recovery,
and a matched extra-compute RSNA-only control. No transfer run or competition
submission was dispatched. Accepted RSNA masks/folds/base and protected
confirmation/gold remain unchanged and unused.

Separate owning-node read-only CPU867932 verified each of the ten original
sidecar training rosters equals the original source admission list exactly
and is disjoint from all119validation exams. It reused existing checkpoint
byte proofs. Exact observer code and sanitized actual receipt are archived
in original-roster-reconciliation.py.txt and all10-original-rosters-verified.json.
