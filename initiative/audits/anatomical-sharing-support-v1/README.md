# Share anatomical features without inventing disease labels

Actual fitting-only projection of the original2247 studies, bound to the live
875858 CPU admission and original release/folds, found:

| Logical auxiliary task | Observed positive | Observed negative | Unknown |
| --- | ---: | ---: | ---: |
| ACL or MCL | 176 | 1505 | 566 |
| Medial or lateral meniscus | 996 | 879 | 372 |
| Contusion or fracture | 124 | 372 | 1751 |

MCL still has exactly **five** observed fitting positives and fracture **three**.
These unions do not create any additional rare-disease labels. They supply
denser, explicitly defined auxiliary tasks that could train shared anatomical
representations within one MRI-only DINOv3 model. A group is observed positive
if any observed member is positive; it is negative only if **all** members are
observed negative. Other combinations remain unknown. No claim of complete
clinical truth or population prevalence follows from report-derived labels.

There are117 fitting cases with observed MCL-negative/ACL-positive status,
533 lateral-meniscus-negative/medial-meniscus-positive, and12 fracture-negative/
contusion-positive. These are crucial specificity controls: **copying a sibling
or group score into the rare head is not justified**. The meniscus union has
988 distinct positive leakage components; ligament and bone have176 and124.
All original component separation and fitting/outer membership hashes match.

## Research decision

Admit the support evidence for a future **shared anatomical representation with
disease-specific residuals**, retaining distinct12 outputs. The proposed
mechanism would learn a shared tissue/view query from the logical group task,
and a separate disease-specific residual so it can distinguish sibling-positive
negatives. This is an inductive-bias hypothesis, not a new independent dataset,
teacher, ensemble or demonstrated accuracy gain. Existing joint encoder training
already permits some transfer; the next comparison must establish whether
explicit sharing adds anything beyond that.

Do not implement a permanent label overlay or a forced probability hierarchy.
Do not tune it using the two MCL development positives. A future paired fit needs
the same initialized graph/RNG, MRI presentations, encoder/optimizer budget,
original labels/masks/folds and a fixed endpoint. Report all12 target and macro
paired whole-component uncertainty, plus the prespecified observed
sibling-positive/target-negative specificity subsets. A gain driven only by
rare-target instability is insufficient for promotion. Repeat a promising result
on an independently matched seed/fold.

The mechanism has **not** been integrated, GPU-admitted, trained or submitted.
Healthy875858 and queued877134 remain unchanged. Complete those architecture
comparisons first; use their actual outcomes to select the representation on
which to test sharing. No extra GPU allocation or checkpoint/publication
funding is supplied by this CPU metadata audit.

`scripts/research/anatomical_sharing_support.py` reuses the accepted fitting-only
projection, verifies the live2247/1156 roster bindings before counting, and emits
only aggregates. Outer, protected confirmation and gold target values are not
decoded. Five tests exercise unknown/NaN masking, binary-policy rejection,
non-mutation, leakage-component counts and sibling-specific negative counts.
No MRI bytes, encoder/model/optimizer or GPU ran for this audit.
