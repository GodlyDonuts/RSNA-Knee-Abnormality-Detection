# Full source-zero paired-view candidate

This implements the accepted next experiment: all3,887 fitting studies, all
73,136,096 parameters, original parent-only teacher, labels/masks/proposals/order,
two anatomical views, effective batch8, fixed3epochs/1,458updates. The accepted
engine14f4b234 is unchanged. The existing mixed-source and H100 ranking runs are
untouched. This is an accuracy hypothesis, not a demonstrated improvement.

Actual two-T4 recovery139736683 passed complete full/stop1/resume state, snapshots
and15x12 two-view full94 CSV equality. Actual CPU input139738563 passed32studies/
64views/image-target-RNG equality and measured3.244x CPU case-loading speed.
The compiler consumes those exact original private sources/receipts and the
unchanged admitted reader. Local tests cover continuous source/worker bindings,
refusal of wrong prior checkpoints, full AdamW and source-zero publication checks.
They do not replace the actual GPU/input acceptances or establish accuracy.

Epoch sessions preserve one continuous optimizer/RNG/LR schedule. Every actual
epoch must be independently CPU-verified and privately published/downloaded/
size/SHA checked before a later session resumes. Three private models plus
optimizer states and verification space have a new4GiB node-local cap, with a
separate2MiB control cap,96KiB ledger allowance and zero new permanent shared
files. All old liabilities and20GiB/5,000-file reserves remain funded. Originals
are never deleted. A byte-verified previous full-model snapshot supplies only
the independently checked graph shapes/dtypes, never initialization weights.

Each new GPU session is capped6,000seconds and rechecks authoritative free quota,
capacity and canonical claim. Only the fixed-final checkpoint is planned for
evaluation; its3,000second serving reservation matches the existing per-epoch
policy (qualification plus bounded public example). This does not pre-fund three
future GPU sessions or authorize paid credits. Partial epochs aren't a final
result. Hidden competition timeout remains9hours in the later serving route.

The manifest is copied into an owner-only ephemeral `/tmp` file, not exported in
Kaggle outputs. This prevents duplicate matching manifests when the previous
epoch's outputs are attached for resume. Every selected window still matches
the complete original integrity proof. No pixel/weight/target Dataset is newly
created. Model/code permissions and exact source/runtime/fold provenance follow
the original accepted sources. No diagnostic or training CSV is submitted.

T4 sequential/DDP grouping and the H100 joint/one-rank pointwise control differ;
a later score difference is not a strict source-weight-only causal comparison.
Protected gold/confirmation remain unused. Actual competition evidence decides
usefulness. Existing intents or uncertain mutations require read-only
reconciliation; no automatic restart, reupload or submission retry.
