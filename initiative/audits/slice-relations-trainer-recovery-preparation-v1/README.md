# Actual production-trainer recovery pilot: preparation only

The new scripts/newton/probe_relation_trainer_recovery.py is prepared and locally
tested. It has NOT run on H100, and no new Slurm pilot/control/source pack or
production relation job was staged or dispatched. Gradient diagnostic866846 is a
separate already-submitted immutable4acd6fa job, still pending afterok:866716 at
this preparation. Never duplicate it or change either sampling source7ac7a97.

## Why this is the next gate

The completed32-pixel FP32 CPU geometry-only probe proved its own two-step
recovery. The queued224-pixel BF16 gradient measurement will measure real
classification/relation gradients without optimizing. Neither proves that the
actual production trainer can optimize the combined objective and recover on a
real H100. This new orchestrator calls the real rsna.competitive.train CLI in
fresh independent processes; it does not reimplement its optimizer or loss.

All three original zero/physical/shuffled pilot configs remain unchanged:
16 actual accepted training studies, two epochs, batch1/accumulation8, official
DINOv3-L/rank32/alpha8,224 pixels/six series/16 uniform slices and four effective
updates per trajectory. For each arm it runs a full reference, an interruption
after update1 (expected exit75), then a separate resume to the fixed end
(expected exit0). No reference/pilot checkpoint initializes production.

## Input and gradient gates

The exact original CPU input receipt, frozen sampling-run binding, release/folds,
16-study training tensors and complete original cache envelopes are rechecked.
The GPU gradient receipt must be terminal/complete, bound to the same CPU
receipt/configs/tensors, finite/nonzero, and have identical classification
rosters/gradients/logits across controls. An explicit decision must bind the
actual gradient receipt SHA, unchanged numerical configs and a written
training-only review of objective scale before any pilot launch. This is an
agent engineering review, not a new user approval requirement. No decision is
present yet because the actual GPU measurement has not run.

The real trainer also constructs and predicts the first16 accepted outer-fold
studies, disjoint from the first16 training studies. Those original bytes must
be separately CPU-admitted/materialized before allocating a GPU. This is a
32-study engineering fixture, NOT a competitive evaluation cohort. Its tiny
AUC outputs must never select a model/recipe, be called competitive results or
be submitted to Kaggle. The proper clinical comparison remains the fixed-final
1156-study outer fold, followed by another matched fold/seed if promising.

The new standalone orchestration source must be separately Git-packed and hash
bound. The original4acd6fa input/source manifest remains the reference for the
CPU/gradient receipt and all rsna numerical modules. A future launcher must
verify every original core module byte against that reference, plus the new
standalone entrypoint/source revision. Do not silently substitute an expanded
manifest into the old CPU receipt or alter its frozen control. No materializer
or GPU launcher for this new32-study fixture is staged yet.

## Acceptance contract

Compare every decoded final checkpoint value: all learned model tensors,
immutable-base metadata, AdamW state/groups, LR, counters, Python/NumPy/Torch/CUDA
RNG and complete numerical/code/data binding. Pickle file hash equality is not
substituted for decoded-state equality. Compare all four objective/gradient/
observation records across full versus stop/resume, excluding runtime/memory
which can differ. Require the exact0:1 stop and2:0/step4 terminal boundaries,
actual source module hashes, identical16-study/12-target MRI-only CSV bytes,
finite probabilities, exact memberships, pinned encoder provenance and BF16
CUDA binding. Numerical or prediction differences reject recovery.

The recorded full-training elapsed interval and peak allocation cover a tiny
four-update diagnostic, including checkpoint overhead; total time also includes
recovery and inference. Neither is a full-campaign throughput prediction. The
script records zero clinical-accuracy evidence and production_candidate=false.

## Resource and preservation requirements

Do not allocate a third GPU or displace either live sampling role. After the
actual gradient result passes, prepare a distinct funded32-study CPU admission,
then the one-H100/two-CPU bounded trainer pilot. Runtime/source/pixels belong on
node-local scratch, with their own reserve. Fresh shared-storage admission must
fund every retained diagnostic boundary, reference/replay epoch, publication
verification and all native/sampling obligations while preserving20GiB/5000
files. Funding from the earlier zero-checkpoint gradient diagnostic does NOT
fund these new actual optimizer checkpoints. No new quota reservation or GPU
submission is implied by this preparation. Do not delete originals for space.

A successful pilot still leaves full-campaign funding, a separately bound
private relation-epoch publisher and exact offline deployment acceptance before
production or competition submission. Preserve actual finite optimizer progress,
negative results and exact source/license/weight/environment provenance.

Thirty-one local checks exercise decoded-state mutation rejection, loss/control
admission, exact original source and MRI output membership, nonfinite rejection,
real0/75 subprocess contracts, and no retry over an existing invocation log.
These are small CPU/synthetic fixtures, not actual official-weight GPU recovery.
