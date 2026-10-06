# Review of research PRs3 and4

The most useful change of direction is a matched public-versus-accepted-v2 label
source experiment. Do not assume our report labels are superior for the competition.
Preserve the exact accepted release/evaluation and change only fitting supervision
within the same strong rank32/native384/full-length recipe. The PR overlay loader
already has useful full-roster/release/fold/leakage-component/hash safeguards.

Neither PR was merged. Original healthy/pending training sources were untouched.
Reviewed exact heads and numerical reproductions are in verified-review.json.

## Blocking or material findings

1. PR4 loss.py77–79 computes BCE(logit,0.5), and train.py949–956 SUBTRACTS it.
   This maximizes cross entropy, not prediction entropy, and pushes masked cells
   toward certainty. A real CPU autograd step moved logit2 to2.001904 and probability
   .880797 to.880997. Implement actual Bernoulli entropy with the correct sign, or
   minimize the0.5-target BCE; add a gradient-direction regression before training.
2. All26 emitted configs freeze DINO and omit encoder.adapter. Their encoder has
   zero trainable parameters under configure_encoder_training. They revert to
   classifier-only fitting instead of our10.2236M adapter recipe. Rebind configs
   to the current accepted model/batch/LR/precision/schedule and actual roster.
3. build_blended_overlay.py rebuilds retained v2 targets from __state. Comparing
   its2247 fitting-study artifact against actual accepted values/masks finds418
   changed cells on the eight targets claimed unchanged. Copy accepted target
   values/masks for retained targets; a source-blend contrast must isolate its
   intended four target changes. No protected label rows were read in this audit.
4. Gold58 was used by the PR author to choose target sources, change severity
   rules and select pass2. Keeping those rows outside optimizer fitting does not
   restore independent evaluation. Treat gold-informed policies as exploratory,
   and do not validate/promote them on the same gold. The gold table's recalls
   also use incompatible coverage: Synovitis has27 gold positives but only15
   v2 decisions, so full-cohort recall cannot be1.00 (upper bound15/27). Report
   same-denominator sensitivity/coverage and uncertainty before declaring winners.
5. data.py217–218 flips every plane horizontally by study side. DICOM physical
   left/right is not every image's column axis; for sagittal IOP row=(0,1,0),
   column=(0,0,-1), it reverses anterior/posterior instead. Use actual orientation
   to mirror the relevant image axis or slice order and transform positions too.
6. The SSL manifest builder joins the entire inventory without a fitting-fold or
   protected/excluded-study filter. Membership-only audit confirms58 gold-evaluation,441 weak-validation and4
   quarantined studies are included. No target columns were read. An819k-slice
   manifest is not clean held-out training admission. Build fitting-only leakage-component-separated manifests
   before claiming independent validation, and admit pixels/runtime separately.
7. EMA updates all teacher parameters, including frozen base, and serializes
   teacher.state_dict(). In10000 identical FP32 student/teacher scalars one step
   changed1549 values by rounding; avoid updating frozen base and store only the
   changing teacher state. Dense DINO-L teacher snapshots also need new storage
   funding; exact tiny-model resume does not cover production byte obligations.

## PR3

Rank fusion is sensible future infrastructure and explains thirds in three-row
example outputs. It does not improve a solo model, and combining correlated members
can worsen AUC. The ensemble/public-label explanation is plausible, not a proven
root cause; attached label tables do not prove every member's training source.
Frozen-global-CLS epoch plateaus do not establish the limit of adapted DINOv3.
Keep rank fusion deferred under the owner's solo-model-first objective.

## Next action

Prioritize only the plain public-label source versus accepted-v2 comparison first.
Rebuild fitting-only overlays from the accepted values/masks, retain the same
2247/1156 rosters and untouched evaluation, use the actual rank32 recipe and fixed
ten epochs, and compare all12 targets with paired whole-component uncertainty.
Use original labels as control; do not incorporate gold-selected tiers/blends,
mirror-all-planes normalization or the broken entropy objective. Complete the
currently committed baseline and cross-sequence comparisons unchanged.

Official geometry source: [DICOM Image Plane Module](https://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.2.html).
