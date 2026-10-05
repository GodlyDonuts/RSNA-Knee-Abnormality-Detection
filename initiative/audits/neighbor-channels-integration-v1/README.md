# Matched neighbor-channel canonical input

The original endpoint-heavy distant control was confounded (275unique slices
versus419for physical neighbors). The new deterministic control permutes the
physical donor pairs between sufficiently distant centers within each series,
preserving both donor multisets and the exact union of observed original slices.
If no admissible rotation exists, both arms use center fallback. This controls
the raw pixel population, not its conditional relationship to the center.
The original confounded source and negative observation remain untouched.

Canonical loader, trainer/config/checkpoint and inference now support an explicit
optional neighbor policy. Default and strength-zero paths retain original gray
bytes; nonzero arms send bounded three-channel inputs through the same official
DINO normalization/learned topology. No base-weight change or extra encoder pass.
Labels/masks/folds, series selection, center views and model attention stay fixed.
Other experimental mechanisms cannot be combined in this first input contrast.

Actual one-CPU/zero-GPU872158 completed on evc31: fitting-only12,476series across
2,247studies,255pixel views from16studies, matched419/419unique physical/distant
slices. It also exercised1,343selected views through the actual canonical loader
and proved byte-identical zero input against the original C9 loader. Non-image
features, positions and masks remained equal; source/cache stats and original
whole-byte proofs were bound. No labels or outer pixels, model or GPU execution.
Original cache/runtime reused; no remote files written.

Synthetic canonical full/stop1/resume tests pass complete model/AdamW/LR/cursor/RNG
and exact CSV equality for gray/physical/distant. A reduced-depth source-topology
checkpoint restoration test covers the color input contract. These are software
tests, not official-weight real-MRI GPU recovery or accuracy.

The separately prepared GPU recovery orchestrator uses fresh real trainer
processes,16fitting/16outer engineering studies,2epochs/4updates and the exact
official frozen base/rank32 graph. It is not dispatched by this CPU receipt.
Owning-node source/input/wrapper/funding admission must precede reservation;
all diagnostic checkpoints remain private and cannot be submitted. Proposed
node-local funding is8GiB total (6GiB checkpoints,1GiB caches, source/metadata
margin), preserving20GiB free; zero new shared bytes/files. No production fitting
or clinical improvement is claimed.
