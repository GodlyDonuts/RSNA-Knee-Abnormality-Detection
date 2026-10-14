# Recent strong single-model evidence and implications

Read October 13, 2026. Primary references:

- [Danial Zakaria, single CoAtNet 0.949](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792).
- [Original associated NDA methods record](https://nda.nih.gov/study.html?id=3407).
- [Tucker Arrants, implementation and validation advice](https://www.kaggle.com/tuckerarrants/discussion).
- [Pilkwang original v15](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1).

The CoAtNet author reports a single learned checkpoint (selected epoch-weight
averaging), dense triplet bags, target-specific attention, anatomical mirror
inference, and mixed weak supervision. Its final recipe also uses external OAI
examples. These are author-reported pipeline outcomes, not our reproduced score,
a clean resolution-only comparison, or evidence of external-data eligibility.
The author's competition-only starting recipe is already strong. Consequently,
ensembling and hardware alone cannot explain our deficit. Repeated tuning on the
small annotated set is declared; it is not independent validation.

Our existing DINOv3 classifier ALREADY has target-specific slice/series attention.
The source-slot candidate now running engineering recovery instead reproduces the
older DINO recipe's crop, grouping, CLS/patch features, augmentation and slot head.
That original v15 public page reports 0.891; matching it is not reproducing 0.95.
Its window averaging differs from the newer complete-bag disease attention.
Earlier native mean-patch fitting regressed; this new bundled recipe is not proof
that patch means alone help. Preserve that negative result.

## Next falsifiable questions

1. Complete the already-funded original source-slot recovery and fixed-final fit.
   This establishes a source-faithful representation/sampling reference without
   disrupting healthy jobs; no score improvement is claimed before measurement.
2. Test learned target-specific pooling across the window bag against fixed
   probability averaging, on identical source inputs/encoder/supervision and a
   fixed fitting budget. PR38 already contains the bounded focal-window proposal;
   review and integrate that existing work instead of creating a duplicate issue.
3. Separate encoder adaptation strength from input/head changes. Original DINO
   members fine-tune multiple blocks with a different backbone LR and schedule;
   the canonical dense-prefix candidate is prepared but not GPU-accepted or fit.
4. Evaluate epoch-weight averaging from predeclared training epochs with a fresh
   matched control; do not pick weights by repeatedly probing protected labels.

The public teacher collection currently measures predictions, not independent
truth: unknown training membership and correlated members prevent clean teacher
validation claims. Source models are not automatically eligible student labels.
No new external acquisition, live recipe change, competition submission or
accuracy result was produced by this research update. Preserve model/slot/input
source versions and distinguish complete recipes from isolated causal effects.
