# External OOF repair negative: code audit
14 October 2026, 20:18 UTC

## Decision
Keep the proposed one-shot constant-total-weight teacher blend. This repository supplies an author-reported negative for a related, state-gated self-distillation recipe, not a clean test of our fixed global blend or evidence for a numerical supervision ceiling.

Pinned GitHub default-branch commit: [1c386ac71385ba7683f550b87a465386ba6c36a8](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/commit/1c386ac71385ba7683f550b87a465386ba6c36a8), dated 27 September 2026.

## What actually changes
Let Q be this author's Opus-4.5 report probability and T its original Arm-A OOF probability. The implemented target is:

- Synovitis, report state u, finite T: 0.40Q + 0.60T
- Effusion, report state u, finite T: 0.50Q + 0.50T
- Every other cell, including explicit positive/negative states: Q

There is no positive-confidence threshold, disagreement gate, ranking transform, or ensemble teacher in the repair function. Every eligible silent cell is mixed, including low teacher probabilities. Missing predictions retain Q. Duplicate OOF IDs are rejected, but this alone does not establish training exclusion. [Repair code, lines 58–120](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/blob/1c386ac71385ba7683f550b87a465386ba6c36a8/tools/repair_labels.py#L58-L120)

The trainer uses dense, unweighted mean BCE across all twelve targets. Silent cells already contributed before repair; no supervision mask or sample-weight expansion is implemented. Total cell weight therefore stays constant, genuinely sharing one feature of our proposal. The state-dependent and target-specific mixture does differ. The six-slot image-presence mask is an attention mask, not a label mask. Their underlying report targets are a separate extraction recipe, not our accepted label blend. [Trainer loss](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/blob/1c386ac71385ba7683f550b87a465386ba6c36a8/train/train_armA.py#L424-L440), [extraction prompt](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/blob/1c386ac71385ba7683f550b87a465386ba6c36a8/sql/02_prompt_v1.sql)

## Model, population, and selection
The intended teacher is five original Arm-A models: RadImageNet ResNet50, slice then slot gated attention, 224px, eight slices per six slots, eight epochs. Random study folds use seed 1234 over all available labeled/cache-matched studies; the README reports 4,407, including the 58 gold studies. Gold targets do not enter BCE. The repaired student uses the same Arm-A trainer; the pusher's repaired switch changes only the label filename unless other CLI options are supplied. This supports nominal matching, not proof that the historical runs used identical configurations. [Trainer](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/blob/1c386ac71385ba7683f550b87a465386ba6c36a8/train/train_armA.py#L350-L414), [pusher](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/blob/1c386ac71385ba7683f550b87a465386ba6c36a8/tools/push_train_kernel.py#L28-L55)

Three independence/selection limitations matter:

1. Repair targets were chosen using calibration-gap and image-versus-report diagnostics on those same 58 gold cases. The code commentary explicitly reverses earlier PF-OA/Effusion choices using them. [Selection rationale](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/blob/1c386ac71385ba7683f550b87a465386ba6c36a8/tools/repair_labels.py#L10-L42)
2. Code-derived consequence: a globally generated five-fold OOF table is not nested outer-safe for five-fold student retraining. For student validation fold f, training rows from fold g use teacher −g, which trained on f. Individual-row OOF does not protect the student's validation boundary.
3. Checkpoints maximize validation AUC against the active CSV's probabilities thresholded at 0.5. Repair changes that selection endpoint as well as training targets. Gold AUC is logged each epoch when enough gold cases exist, but does not drive the automatic save condition. Human selection from those logs cannot be established. [Selection code](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/blob/1c386ac71385ba7683f550b87a465386ba6c36a8/train/train_armA.py#L454-L490)

## What the numbers establish
Gold macro .8955→.8891, Synovitis .840→.802, Effusion .957→.942, and 3,803/4,407 Synovitis cells rewritten are README author reports, not independently bound run receipts. The staging script repeats the macro comparison but provides no repaired-run identity or aggregate evaluation artifact. The complete repository tree has no standalone training/evaluation receipts; output directories and logs are ignored. No CSV was opened to recompute results. [README](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/blob/1c386ac71385ba7683f550b87a465386ba6c36a8/README.md#out-of-fold-label-repair-made-things-worse), [staging comment](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/blob/1c386ac71385ba7683f550b87a465386ba6c36a8/submit/stage_checkpoints.py#L1-L29), [ignored outputs](https://github.com/NTejas-1/RSNA-Knee-Abnormality-Detection/blob/1c386ac71385ba7683f550b87a465386ba6c36a8/.gitignore#L10-L37)

The .941 claim concerns a large reference ensemble plus ten Arm-A/B checkpoints, not the repaired single model. Do not accept its universal correlation gates, .003 noise floor, certainty of gold ordering, or causal explanation of failure. Our experiment still needs admitted teacher ancestry, an unchanged fixed endpoint, and outer-safe target production; globally cross-fitted targets alone do not satisfy that boundary.

Scope: public source/aggregate commentary only; no clinical rows, reports, images, weights, training, inference, submissions, or external mutations.

