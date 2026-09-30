# Exact delta recovery and whole-series production launch

September 30, 2026. Scientific source is
`c9fcc7eefb09103942be89bf4fa87410eb9ec5ab`. Source checks passed Ruff,
formatting, mypy (38 modules), 1,122 tests and accepted-data archive/coverage
checks. The role-aware publication coordinator and rejection tests bring the
verified suite to **1,136 passing tests**. Current v2 coverage is 89/89 complete.

## Actual recovery and preservation

Bounded Newton **865346 COMPLETED 0:0 in 18:42 on evc28**, one H100/two
CPUs. Full-storage reference, expanded delta, and deliberately interrupted then
resumed delta training have identical model tensors, optimizer, RNG, progress
and prediction bytes. All 120 adapter modules changed; all 415 original frozen
base tensors remained unchanged. Reconstruction and inference on real MRI data
from the verified official base plus delta also passed. See `terminal/`.
The resumed torch archive has different serialized bytes; its decoded payload
and predictions are exact. This is a recovery proof, not an accuracy result.

The 126,363,429-byte schema-2 diagnostic and 19 accompanying objects were
privately published once and fully re-downloaded/size/SHA verified at HF
**111f78f91f37537564f8c109904511e501ee2757**. It requires the pinned original
base and remains diagnostic-only. Its SHA is
`5d1b3b562160e803442d33316779401a4fa4721370b182187802b14c78a1c22e`.
Do not repeat its job, dispatch or publication. The earlier whole-series compact
pilot remains a distinct non-resumable historical diagnostic. Current Kaggle
packaging rejects schema 2 until separate offline deployment acceptance.

## Full held-out control is running

Latest completed **epoch 3 / 843 updates** has all four private objects fully
re-downloaded/size/SHA verified at HF
**a5e39b7332a8d0bd5be06e474fcea6219e90c421**. Epochs 1–3 are preserved;
never repeat them. At 12:35 UTC the job was finite in epoch 4 at 884 updates,
with empty stderr. The fixed-final held-out result remains outstanding.
Separate six-MRI private T4 full/delta restoration acceptance has since passed;
see the deployment audit. Each production candidate still needs its own
source-bound offline and hidden-scale runtime checks.


Full production **865357 RUNNING on evc28**, one H100/two CPUs/64 GiB RAM,
48-hour bound, was dispatched ONCE after exact recovery and private preservation.
Allocated driver, actual CUDA context and full cache/source/environment preflight
passed. Real finite optimizer progress is recorded in `production-start/`. Source checkout:
`/home/sa305415/rsna-knee/code-delta-recovery-c9fcc7e`.
Control: `/home/sa305415/rsna-knee/production-launch-whole-series-control-c9fcc7e-v1`.
Unique output:
`/home/sa305415/rsna-knee/runs/dinov3-l16-wholeseries-control-a32-fold0-c9fcc7e-production-v1`.
Never duplicate this dispatch or reuse its output.

This control uses all cached slices from all accepted series, unchanged labels,
rank-32 adapters, no neighboring-slice context, 2,247 training studies and the
same 1,156-study outer holdout. Ten epochs, 281 updates/epoch, effective batch 8.
Compare its fixed final result to the sampled A32 reference (0.8620940794 internal
macro AUC). No current competitive metric exists for this control. Inputs remain
224 pixels from the 256 cache; this does not restore native resolution. The
matched neighbor-context challenger is prepared only and has no production job.

For each completed unattempted epoch, use
`scripts/hf/dispatch_whole_series_epoch.py N --variant control --ssh-control SOCKET`.
Pinned external publisher SHA:
`7ff5db4f1cd6b0b338f6244ff78375bd814c93cd72e3bc8e0eb11573c036aa02`.
It rejects diagnostic/sampled/all-data/wrong-role/wrong-base/changed-code payloads
and fully verifies the four private objects. Existing intent/journal means
read-only reconciliation; no blind retry. The checkpoint stores learned tensors
and complete recovery state against the immutable verified base.

## Storage and independent work

CPU retention **865354 COMPLETED 0:0** removed only privately verified primary
epoch-2/3 redundant files and redundant already-submitted A32 Kaggle checkpoint
and ZIP copies. The original trained final, active primary checkpoint, all
sidecars/receipts, private HF bytes and immutable private Kaggle Dataset remain.
The first retention harness failed before deletion because `/home` resolves to
Lustre; read-only reconciliation proved all four files unchanged. V2 uses exact
paths confined to the resolved base. Both attempts and terminal evidence remain.

Fresh production projection leaves **20.005379 GiB** after all remaining primary
full checkpoints including atomic writes, its final bundle/ZIP, ten control delta
snapshots plus live/atomic files, three future control deployment copies and
132 MiB metadata/source allowance. File reserve: see exact storage precheck.
This is tight; recheck both quotas before any transfer/new job. No third GPU
node, no large local download, and no unverified retention are authorized.

Primary all-data **865263** stays untouched and healthy in epoch 4. Epochs 1–3
are each privately fully verified; epoch 3 revision
**8e1a49f3d28e49b449b83b53f73fd7863dadcd98**, 1,278 optimizer updates.
Do not republish them. Exact Kaggle **56697342** remains pending/unknown with
null status/error/score and four live slots at **07:55 UTC**; its canonical ledger
is `submitted`, exact notebook v1. No new competition submission was made.
Use useful, fully accepted completed candidates and authoritative live allowance;
never upload the three-row public CSV or this diagnostic.


The source/recovery/control-launch/epoch-3 small milestone is on GitHub main
`0dba237` and separately privately mirrored at HF
**acc7d4f2a7317b32b13ab473ecfb3755b97f5769**. All 104 objects were fully
re-downloaded/size/SHA verified. `hf-terminal-*` records pin the exact public
source commit and private file manifest; do not repeat this publication. Subsequent
read-only checks observed control update 14 and all-data update 240 of epoch 4,
both RUNNING and finite. These are training progress, not final accuracy.
