# Mission and completion criteria

The owner expanded the mission on 2026-09-22: after the data is training-ready,
investigate, theorize and plan what to train with it to maximize the RSNA Knee
Abnormality Detection Kaggle score. This adds a second phase; it does not replace
or relax data-readiness requirements. Do not treat completion of phase one as
completion of the expanded project.

## Phase one: verified training data

Secure the supplied credentials in ignored local secret storage; recover the
original aggregation and its provenance; validate image integrity, source/report
correspondence and report-derived annotations; resolve semantic review gates;
handle corrupt, duplicate and unsuitable supervision reproducibly; isolate related
studies and evaluation data; and produce final training manifests and a readiness
report backed by actual loader, training, resume and inference checks.

The detailed release checklist remains in [decisions.md](decisions.md), with
current evidence and unresolved work in [codex-continuation.md](codex-continuation.md).
Original sources and earlier judgments stay preserved. Official gold is
evaluation-only; missing or uncertain report evidence must not become an invented
negative. Review coverage, green software tests and checkpoint publication are
necessary evidence but do not independently prove readiness.

## Phase two: competitive training research and plan

Begin the full modeling investigation after phase one passes its acceptance gates.
Use the actual final data characteristics and measured supervision coverage to
produce a concrete, evidence-backed strategy. The deliverable must include:

1. The current official competition metric, inference/submission constraints,
   external-data and pretrained-weight rules, with primary-source references.
2. An evaluation design that separates development decisions from the untouched
   evaluation data, respects duplicate/related-study groups, and measures site,
   language, scanner and target differences where supported by the metadata.
3. A critical comparison of plausible image architectures, study/series fusion,
   resolution and sequence choices, initialization/pretraining, and ways to use
   weak report supervision and its uncertainty. Distinguish sourced findings,
   hypotheses and measured results.
4. One recommended starting approach, alternatives worth testing, and explicit
   reasons grounded in our final data and available compute.
5. A prioritized experiment matrix with baselines, controlled data-quality and
   supervision ablations, success/failure criteria, and rules for choosing the
   next experiment. Test whether cleaning improves generalization; effort alone
   does not establish superiority over other competitors' data.
6. A compute and implementation plan: memory/storage requirements, estimated
   training and inference costs, reproducibility, checkpoints, and experiment
   tracking. State any hardware assumptions rather than silently inventing them.
7. A later-stage plan for ensembling, calibration, test-time augmentation and
   submission selection, with checks against validation overfitting and
   leaderboard noise.

The intended outcome is the strongest defensible route to competitive Kaggle
performance, not a promise of a particular rank. A list of fashionable model names
or untested claims is not a completed strategy. Research should narrow the choices
to an actionable, costed sequence whose assumptions can be tested.

## Goal tracking

Both phases are now required by the user. The goal is active; its status control
cannot edit the original objective text in place. This document records the user’s
expanded scope for continuation and completion audits.
Do not mark the expanded mission complete until the data-release evidence and
the competitive training research/plan have both been delivered and verified.

## H100 and CPU-cluster expansion

The owner subsequently confirmed access to an H100 GPU server and a large CPU
cluster, with access to follow once the plan is agreed. Revise the modeling plan
for scalable multi-GPU experiments, native-resolution preparation and MRI-specific
representation learning. Critically compare DINOv2 with newer encoders and 3D
approaches; popularity in public notebooks is not evidence of optimality. The user identified UCF Newton, with opportunistically available two-GPU nodes
and inter-node InfiniBand. Exact allocation, assigned memory and usable duration
remain to be confirmed at access. Training
resources do not relax the competition’s offline nine-hour inference limit.
The deliverable remains a defensible first-place-oriented experiment strategy,
not a guarantee of winning or authorization to incur new compute charges.
