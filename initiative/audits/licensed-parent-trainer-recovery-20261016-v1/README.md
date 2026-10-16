# Exact-parent end-to-end numerical engine and real-MRI recovery wrapper

Prepared on October16 after lossless original MRI inputs were actually admitted.
This is not a new accuracy result, actual GPU acceptance or full-fit dispatch.
External MRI foundation/source493c7b0d/weight7e5315da stay attributed and private;
no independent localOOF is possible because upstream development used local
cohorts. The unchanged released foundation scored.949; own ImageNet-based N3.894.

The prospective first adaptation is fixed3epochs/3887eligible fitting studies,
1458 optimizer updates, effective8, all encoder+MIL parameters train, uniform12
of the original94windows without donor changes, original anatomical mirror and
view-specific full-parent teacher. BCE target is.25 observed source+.75teacher;
missing source returns its influence to teacher. Labels/masks are unchanged.
Encoder/head peakLR3e-6/3e-5, AdamW(.9,.999,eps1e-8),WD.02,10%linearwarmup/cosine
ending1%peak,clip1. Parent BatchNorm statistics are frozen; affine parameters
remain trainable and original dropout.2 trains. This recipe intentionally
approximates the94window bag during training; the full94two-view path remains
evaluation. Its benefit is an unproven hypothesis, not an inherited.949 score.

The engine supports one/two fixed ranks; twoT4DDP uses FP16autocast/FP32loss and
complete GradScaler state. Actual final odd7-case batches pad with zero-loss
forwards so both ranks synchronize; denominator counts only the7 real studies.
Optimizer-boundary state includes complete model/base,AdamW,LR schedule,cursors,
history and every rank's RNG/scaler. No silently skipped overflow update.

Thirteen engineering tests passed, including three fresh two-process CPU jobs
(full/stop1/resume) with a synthetic BN/dropout graph, changed encoder+head,
complete state equality and preserved original BN buffers. These are synthetic
engineering tests, not MRI training or CUDA qualification. Five compiler tests
passed; Ruff and mypy are clean. Combined cache/engine/compiler tests36passed.

The compiled private real-MRI wrapper preserves the exact original encoder/MIL
and strict weight loader, original32pixel/feature indexes,16fitting+16disjoint
engineering cases,2epochs/4updates, all94two-view engineering predictions. It
requires actual twoT4 complete state and exact prediction bytes, nonzero finite
encoder/MIL gradients, and records actual RAM/VRAM/time. It cannot submit and
contains no independent accuracy metric. Private UIDs/targets/source bodies and
weights are injected only on authorized storage; no clinical rows enter Git.
Actual source/GPU quota/funding and single coordinator dispatch are still
required. Do not treat a prepared wrapper or CPU tests as real GPU acceptance.
