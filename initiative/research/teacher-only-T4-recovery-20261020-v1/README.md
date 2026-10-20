# Paired zero-source supervision: two-T4 route

The cross-teacher epoch-1 entry actually scored **0.948**, below the older own
0.949 adaptation. There is no new accuracy improvement. The running H100 matched
pointwise/ranking experiment remains unchanged.

This separate experiment tests whether the original-label loss is harming
teacher adaptation. Labels, masks, teacher values, proposals, model and optimizer
are preserved; the source loss weight is explicitly zero. This is an ablation,
not evidence that the source labels are incorrect. It uses the original parent
teacher, not the changed cross-lineage target, and must not be described as the
same scientific comparison as the one-view changed-target Kaggle run.

The pointwise paired-view loss is separable, so both T4 ranks can process paired
anatomical views with a global mean over twice the actual study count. The last
one-study batch includes a zero-loss padded rank; only the final microcase's
second view synchronizes gradients. Sampling happens once per study and both
views share the selected positions. The original positive-source recipe and all
live numerical payloads remain unchanged.

Actual local CPU tests exercise fresh full/stop-after-one/resume processes on
17 synthetic training cases, two epochs/six updates, and 15 disjoint synthetic
prediction cases. Both one-rank and actual two-process Gloo runs match complete
model/AdamW/LR/counters/all-rank RNG/scaler/history, epoch snapshots and prediction
bytes. Deterministic one-rank/two-rank normalization is compared through the
final padded batch. These are numerical engineering results, not MRI acceptance
or clinical accuracy. Scoped numerical/source tests: 26 passed.

`engine.py.txt` derives the new loop only from the exact original E024 source.
`compile.py.txt` retains the private original E1 model/renderer prefix and builds
a separate offline diagnostic using `worker.py.txt` and `bootstrap.py.txt`.
`prepare_once.py.txt` only compiles on an owning-node CPU; it cannot launch a GPU.
The real-MRI diagnostic requires two actual T4s, six finite updates, identical
full/resumed state and 15x12 two-view full94 CSV bytes, original FP32 initial
model digest on both ranks, frozen parent BN buffers, and bounded output/runtime.
It deliberately produces no `submission.csv` and is not a competition model.

At the actual 03:58 UTC account check, 22,185.86 free GPU-VM seconds and one campaign
lane were available; four competition entries remained. This is a dated read-only
observation, not dispatch permission based on stale allowance. The first CPU
account observer failed because it expected a remote capacity module that did
not exist; the corrected observer supplied the original capacity source in RAM.
Neither observer made GPU, ledger, Dataset or submission mutations.

Before executing this diagnostic on T4, hold the exact H100 diagnostic 975954
only if it is still PENDING with zero runtime. Do not duplicate a diagnostic that
has started. Fresh canonical claim, free quota and source/runtime/input bindings
remain required. A T4 pass proves recovery on T4, not bitwise equality with the
different H100/PyTorch runtime. Full fitting needs a separate actual acceptance,
budget and checkpoint-preservation decision.

Actual CPU976158 compiled the private302,505byte source. The once-only canonical
controller then accepted **139736683v1 RUNNING**, exact source
`a2174506987fb667932011a40ef225b30b1f09f2000803aaec2dc142e2bb7e23`.
Original975954 was held only while PENDING/zero and remains JobHeldUser.
`actual-push.json` is execution evidence, not a GPU recovery pass or an accuracy
result. `follow_once.py.txt` observes only this version and collects the aggregate
acceptance receipt after actual COMPLETE; it never dispatches production.
