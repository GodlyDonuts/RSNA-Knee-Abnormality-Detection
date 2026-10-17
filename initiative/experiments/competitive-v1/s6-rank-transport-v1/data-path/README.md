# Explicit teacher overlay in the unchanged full-encoder trainer

October17 2026. The target transport now has a separately versioned data path
and **actual synthetic CPU training/recovery evidence**, not a new MRI fit.
[Source](teacher_overlay.py.txt), [17actual checks](checked.json),
[test program](check_overlay.py.txt).

`TeacherOverlay` wraps the admitted original unique-window reader **before** the
unchanged `ProposalCases`. Only `parent_logits` changes. The original reader
still validates original pixels/renderer and chooses the anatomical view;
the sampled wrapper still preserves IID duplicate order and proposal correction.
Source image/label/mask objects pass through unchanged. Targets and metadata are
cloned so caller mutation cannot alter their admitted values.

Production requires the exact3887ordered fitting roster, original parent/source
bundle pins, complete detached finite FP32 N×2×12 targets and old/new numerical
array hashes. The parent-only control requires exact original target bytes and
no S6 teacher. A distinct transported role binds the exact S6 lineage. Synthetic
engineering populations require explicit diagnostic scope and cannot pass the
production population contract.

`fit_with_teacher_targets` automatically attaches the actual data's target
binding to the existing training engine. Callers cannot supply a conflicting
binding or silently resume a different target bundle. No original NPZ, label,
mask, trainer module, numerical recipe or running model was edited.

## Actual test scope

The real unchanged `licensed_importance_train.fit` ran on a nine-study synthetic
MIL fixture with BatchNorm, dropout, checkpointed encoding and a final one-study
batch: two epochs/four updates, effective8/uniform16. Parent-only overlay matches
the original model/AdamW/LR/RNG/history and full94 prediction CSV exactly; only
the explicit new input-binding metadata differs. Both control and transported
roles match **every checkpoint field and9×12CSV bytes** after full/stop1/fresh
model/resume. Changed actual targets cannot resume an old checkpoint, and the
transport changes learned parameters in that same engine. Source masks contain
masked NaNs; original labels/teacher bytes remain unchanged after all fits.

This is **CPU FP32, one rank, synthetic images**. It does not establish new full
MRI input acceptance, two-T4 FP16/scaler recovery, complete fitting-cohort S6
targets, resource admission, clinical truth or accuracy. The already accepted
original two-T4 engine evidence remains separate. Outer admission still needs
actual private full teacher files/receipts/rights and exact source provenance.

Both new data-path programs pass focused Ruff. Complete prior package tests
are reusable only while rsna/scripts/tests remain byte-identical; the original
fitting/inference sources and their existing live jobs stay unchanged.
