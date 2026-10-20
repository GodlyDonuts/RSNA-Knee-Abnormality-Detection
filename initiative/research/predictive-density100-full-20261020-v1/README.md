# Separate full-cohort C100 objective experiment

Prepared October 20, 2026. **No full C100 admission, GPU dispatch, checkpoint or
accuracy result exists.** Original pilot981620 is still pending; the actual
pilot result must be reviewed before this route can execute. The primary
B/C/C0 owners remain unchanged.

## Why this experiment

[CPU981494](../predictive-objective-balance-20261020-v1/README.md) found that the
weighted density gradient at the original C24 state was only0.0102–0.0140% of
the exam gradient across nonencoder parameters on two declared fitting cases.
That motivates one explicit coefficient intervention, not a claim that the
architecture is ineffective or that coefficient100 is optimal. Gradient norms
are not Adam update norms, and this small fitting diagnostic does not establish
generalization.

This branch changes C's density coefficient from0.1 to100. It retains the exact
CoAtNet MRI encoder, fresh predictive heads, parent/source permissions,
initialization, accepted labels/masks, 3,887-study order, seed20261020, effective
batch8/final7, six epochs/2,916 updates, head-only first3updates, and
3e-5 peak learning rate. The architecture remains `arm="C"`; `branch="C100"`
and a distinct source commit identify the new experiment. It is not DINOv3.

The original numerical modules from science44cc559e remain byte-identical.
[derive.py](derive.py) derives the worker and publisher from pinned original
bytes. Existing B/C/C0 scripts, outputs, proofs and private publications are
neither replaced nor reinterpreted.

## Actual prerequisites and execution

[admission.py](admission.py) requires original C100 pilot981620 to finish
COMPLETED0:0 with exact full/stop/resume state and prediction equality. A
separate agent decision must bind its actual receipt, scheduler outcome, all
warnings and the full recipe. Prepared plans and the four engineering-label
predictions cannot supply this decision.

It then waits for the original all-evc27 portable placement to complete and reuses all
3,887 node-local context files by original hash/stat proof. It does not download
or transfer the MRI cache again. Parent/runtime, original input admission,
renderer lineage and common initialization remain explicitly bound; a distinct
C100 admission records the new worker/source instead of relabelling C0's receipt.
The prepared coordinator binds the unchanged `2a224ae0` controller and original
owner32377 intent, together with the actual `7dcd87e3` continuation33107 intent.
It admits the exact original all-three-arm dispatch records and owned log paths.
A lost dispatch reply is handled only through matching sealed-job reconciliation
receipts, never another submission. Portable
input admission must retain the baseline-plus-two-authoritative-files policy
and the actual strict-decoder regression, plus the exact HTTP499 recovery decision and transport hashes in both the original control and its accepted proof map. Stale e9 or pre-correction C0 proofs fail. Original pilot981620, its binding, science44cc and the distinct unadmitted C100 output/funding scopes are unchanged. This retarget does not admit or launch the extra branch.

Private paths on evc27:

- Controls: `/tmp/rsna-predictive-density100-full-20261020-v1`.
- Full output: `/tmp/rsna-predictive-full-C100-20261020-v1`.
- Original inputs: `/tmp/rsna-predictive-evc27-deadline-placement-20261020-v1`.

[coordinator.py](coordinator.py) exposes `status`, `admit`, `dispatch`, `publish`
and `preserve` phases. Each mutation has one durable intent and no retry. An
existing intent requires read-only reconciliation. `admit` first checks actual
prerequisites without writing an intent. `dispatch` requires all three actual
primary jobs and adds `after:C_job:B_job:C0_job`. Every primary must have started
before the optional job becomes eligible; this is not a completion dependency.

The separate full job requests one H100/two CPUs/32GiB/eight hours on physically
two-GPU evc27. New dispatch stops October21 20:00UTC; Slurm's finish deadline is
October22 04:00UTC. These are limits, not guaranteed allocation or completion
times. Both scheduler inspection and submission explicitly use `TZ=UTC`.
The preservation phase continues to October22 noonUTC and never cancels a
healthy original job. The updated optional routes passed123 focused local tests
and independent review. They retain all36GiB of primary fitting scope, the
additional64MiB placement controls, and every optional/descriptor obligation
without speculative paid credit. No operation was launched by these changes.

Example invocation after source freeze and actual pilot review:

```sh
python3 initiative/research/predictive-density100-full-20261020-v1/coordinator.py \
  --source-git <exact-C100-control-commit> --phase status
```

The accepted decision must contain the actual pilot scheduler object, receipt
SHA, source/code hashes, exact `admission.recipe()`, unchanged warning object,
agent reasoning, `engineering_predictions_used_for_selection=false`,
`automatic_recipe_retuning=false`, `primary_B_C_C0_unchanged=true`, and the
unchanged3,360-second primary Kaggle reserve. An absent decision fails closed.
No accepted full-run decision has been written.

## Observe the mechanism through peak learning rate

The24-update pilot reaches only4.94e-6; full fitting peaks at update146.
Passive detached forward hooks observe updates1,24,146,200 and each epoch end,
using the already loaded fitting labels/masks and the existing two forward
outputs. They emit only aggregate posterior change, full-vs-context-only BCE,
BF magnitude and saturation. No extra encoder pass, backward call, data
selection, parameter update or random draw is introduced.

Review fixed windows126–150 and176–200 using existing exam loss,
density-minus-Gaussian-constant, gradient norms and clipping. Warning examples
are more than twice C0.1's same-window exam loss, or sustained strong clipping
with rising exam loss and saturation. These are investigation triggers, not
automatic recipe changes or accuracy selection. Nonfinite arithmetic and
source/state mismatches remain hard failures. Weighted density includes a
constant91.89385; a total loss near92 is not itself a failure.

The fixed-final scientific comparison is C100−C0.1 and C100−C0, with B as
context. Favorable training diagnostics cannot establish the proposed
predictive-evidence benefit. Inherited parent exposure still prevents treating
the local weak-label holdout as independent whole-model validation.

## Storage, preservation and serving

Admission needs fresh capacity for a separate12GiB full-run/publication cap and
16MiB controls/logs, while retaining every original C0/input scope and the
additional6GiB+4MiB C100 pilot liability. Independently retain6GiB+4MiB for the prepared matched G100 pilot and1MiB for the existing descriptor-normalizer artifact, regardless of optional execution order. These two scopes receive no paid-byte credit without a separate actual original proof; reserving them grants no GPU or admission authority. Already physically charged original C0/C100 bytes are
counted once. No originals are deleted; disposable verification downloads never
retire future obligations. All new files are node-local. Shared reserve remains
20GiB/5,000 files after123MiB/376 outstanding files, minus only exact physically
present original GPU logs.

All six completed epochs use a distinct
`checkpoints/predictive-C100-full3887-<source12>/C/epoch-NNN` prefix and the
canonical shared publication lock. Each five-object publication is fully
re-downloaded and size/SHA verified. The single `preserve` phase monitors only
this new job; it does not acquire data, submit models, repeat an uncertain
publication, or alter the primary owners.

The separate [serving contract and offline compiler](../predictive-density100-serving-20261020-v1/)
are implemented. They require actual pilot/full terminals and all six verified
epochs, then preserve the original serving worker, MRI modules, bounded mounts,
descriptor reuse and public-only watchdog. The57 focused contract/compiler
tests establish local engineering behavior, not production execution.
**No actual C100 model has been compiled, no account owner has been admitted,
and no final return, trained T4 qualification, extra Kaggle funding or score
exists.** A later final-only return to evc33 needs its own512MiB capacity; actual
trained runtime/precision and additional account allowance must preserve the
primary3,360-second reservation.
