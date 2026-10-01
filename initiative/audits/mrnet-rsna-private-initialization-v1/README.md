# Stanford checkpoint transfer into the actual RSNA trainer

The owner asked to use Stanford data. The original 3,750 MRNet volumes were
already fully acquired and verified, and bounded GPU pilot 867419 passed
four actual optimizer updates with complete interruption/recovery. Full-source
engine recovery 867512 was submitted once on the original owning node evc32;
its last observed state was PENDING. Full 1,129-exam source fitting has not been
dispatched. This audit adds a downstream initialization path: a source-only
three-task classifier is not our RSNA twelve-task model.

`rsna/competitive/mrnet_initialization.py` loads only an explicitly SHA-bound
checkpoint and terminal receipt. Both must describe the fixed-final full source
fit: original acquisition/input/engine/F12 science bindings, 1,129 distinct
training cases, 119 source validation cases, ten epochs and 1,420 updates. An
engineering trajectory, intermediate checkpoint, different recipe/source,
changed frozen base, incomplete adapter graph, incompatible rank/dtype or
nonfinite tensor is rejected before any destination tensor changes.

The destination must have identical complete encoder provenance and actual
frozen tensor hashes. Only all encoder LoRA A/B tensors copy. A fresh twelve-task
RSNA head, fresh AdamW, RSNA counters and RSNA RNG remain independent of the
source head/optimizer/counters/RNG. Source ACL/generic-meniscus/abnormality labels
never map to RSNA columns. Optional CLI arguments are `--mrnet-checkpoint`,
`--mrnet-checkpoint-sha256`, `--mrnet-terminal`, and `--mrnet-terminal-sha256`;
all four and frozen-base delta storage are required.

Source-derived runs require node-local `/tmp` output beneath an owner-only
directory. Their run/checkpoint binding preserves both source artifact hashes,
original source binding, `private_research_only: true` and
`external_weight_publication: false`. Competition checkpoint admission rejects
these weights even with diagnostic/scheduled-epoch overrides. No raw source
images, label values, private image signatures or learned source tensors are in
this audit. Public competition/derived-weight rights remain unresolved; no
publication or Kaggle admission was added.

Fourteen synthetic CPU tests include transfer into a different twelve-task head,
unchanged base/head/RNG, invalid source/graph/bytes rejection, private-output and
competition refusal, and an actual `_run_locked` RSNA full/stop-after-one/resume
experiment. Its complete final learned state, frozen metadata, AdamW, counters,
all RNG and MRI-only CSV bytes are identical. It starts fresh at RSNA step zero
and finishes at step four, never inheriting the source's 1,420 steps. These are
engineering tests, not real MRI training, an actual full Stanford checkpoint,
clinical accuracy, or acceptance of a new GPU experiment. Full QA: 1,638 tests,
Ruff 258 files, mypy 43 source files; existing importlib warnings preserved.

No existing frozen numerical archive, source recipe or live job was changed.
Before a private downstream experiment, collect actual 867512 terminal recovery,
dispatch/complete full original source fitting once, finish both original pixel
indexes and permitted overlap review, and fund a distinct GPU transfer recovery
and matched experiment. Compare fixed-final RSNA predictions on the same
1,156 development studies against a control with matched additional adaptation
compute, identical downstream recipe/initial head and accepted masks/folds.
Source validation AUC cannot replace this RSNA comparison. Protected confirmation
and gold remain unused. No benefit is assumed from external data alone.

At 23:21 UTC Newton's private DNS was still unavailable. Cisco read-only state
actually reported Disconnected. Native production 867367 and index 867494 remain
last-seen RUNNING; Stanford recovery 867512 is last-seen PENDING. Observation loss
is never terminal/restart evidence. No duplicate dispatch, transfer, upload,
deletion or competition mutation occurred. Latest metadata/code is not yet
HF-mirrored because the shared Newton publication lock is unreachable; do not
bypass it or claim private preservation.
