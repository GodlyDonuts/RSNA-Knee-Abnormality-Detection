# Actual repeated-control evidence and future compute decision

The two healthy original comparisons independently fit a CLS/zero-patch control and a disabled-public-label control on the same original2247 fitting studies. A coincident step record prompted a bounded read-only comparison rather than assuming different source hashes mean different learning.

Actual evidence:

- All six numeric fields (epoch, update, optimizer_steps, loss, gradient_norm, observed_cells) are exactly identical across the first**1410 contiguous shared updates**. Elapsed time and allocated-memory fields were excluded, and they differ. The baseline had2810 observed updates while the public-label control had1410 at collection.
- Independent original CPU881035 compared the two already privately preserved epoch5 checkpoints: **all258 common learned tensors are exactly equal**, their frozen-tensor proofs and encoder provenance are equal, and the baseline's sole extra learned tensor is an all-zero131072-parameter `patch_summary.weight`.
- Original checkpoint byte proofs were reused with sidecar/size/unchanged-stat checks; no original checkpoint download or bulk rehash was repeated. No MRI, labels, protected records, clinical inference or GPU was used.

This is strong evidence of repeated common control work, but **not whole serialization equivalence**: graph/bindings differ, and complete optimizer/RNG state and final predictions have not been compared. Original jobs and scientific recipes remain unchanged. Do not cancel, skip, restart or replace a healthy arm on this receipt alone.

## Agent research/compute decision

For subsequent compatible input/readout hypotheses, design a shared-control comparison prospectively rather than independently rebuilding the same control for every candidate. Admit reuse only when the original initial/effective learned state, data/fold/order, precision/runtime, LR/optimizer schedule, endpoint and resulting relevant state/predictions are bound and compatible. A changed source hash alone neither disqualifies reuse nor certifies equivalence; inactive graph additions require explicit accounting. Preserve every original role, uncertainty and source binding.

The current two runs were dispatched separately before this equivalence audit. That produced avoidable common fitting work. Avoid repeating it in the next experiment design, while preserving these original runs for their predeclared results. A shared control is a compute-efficiency decision, not a demonstrated clinical gain or permission to tune against protected data.

Numerical log and checkpoint proofs are in [actual-log-comparison.json](actual-log-comparison.json) and [actual-checkpoint-comparison.json](actual-checkpoint-comparison.json). Future actual full-state/final-prediction acceptance remains required where applicable; do not describe it as completed.
