# RSNA: decision brief and falsifiable next moves

Research snapshot: 14 October 2026. Historical run ledger pinned to `8300cfd20c2acaeef9d684a81dbc7fe977282a62`; newer public-source observations are dated separately. This document proposes experiments. No experiments were run for this research.

## Working conclusion

The most actionable lead is **better image-derived supervision with adequate anatomical coverage**, not the novelty of the backbone. A small DINOv2 model and ConvNeXt have author-reported single-model scores above .95, so the current .872 DINOv3 result cannot establish a transformer-family ceiling. Those author reports are not independent reproductions or evidence that changing one component will reproduce their scores.

A useful strategy has two tracks: (1) complete the unresolved experiments and evidence already paid for; (2) test one auditable image-supervision intervention against a fixed control. Keep a strong compact-backbone recipe as a practical challenger if its source, runtime and validation conditions can be established. Do not launch a large collection of novel modules.

## What the external evidence actually establishes

- [colum2131, October 8](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3535687): reports .953 ConvNeXt-T and .951 DINOv2-S, both with image pseudo-labels and no external data. Exact checkpoint, TTA and training-reproduction evidence remains incomplete
- [Archit, September 16](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3525138): reports .950 single-fold CoAtNet. [September 20 explanation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3526342) attributes improvement to soft OOF image/report-label blending; hard replacement and masking did not help. Final mixing weights and all fold details are not disclosed
- [Counterexample](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/735304#3529062): a participant reports stronger local/gold CV without public improvement from soft bootstrapping. This prevents treating the mechanism as universally effective
- [nartaa's public .949 notebook, version 2](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror?scriptVersionId=356175397): one SWA checkpoint with anatomical-mirror inference; the Kaggle score badge is stronger evidence than an unsupported comment. It uses MRI-only inference. OAI-derived pretraining, repeated gold selection, and downstream terms must be disclosed
- [nartaa terms](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007): the October 13 update explicitly permits adaptation and noncommercial redistribution with attribution, including the named winner license. This is author-level permission, not a final host ruling on every upstream obligation

## Do not repeat the wrong experiment

Our full-public overlay improves internal macro from .8786706 to .8997841. Public values on old masks score .8536813, and coverage-only scores .8693023. These used a pinned public label table whose exact identity is established by archived SHA-256 receipts: the Shio-Koube copy is byte-identical to Steven Lee’s Kaggle version 6. They are not an executed trial of freshly generated, fold-safe image-teacher soft labels.

The archived source-card findings identify report-LLM parents (author v2 and lixin73 GPT-5.6-Sol), CC0, gold-guided choices and an effusion-to-synovitis proxy. This is stronger than filename similarity. It establishes a reason the 58 gold cases are not untouched confirmation of this source; it does NOT establish image-teacher exposure to the 1,156-study outer cohort. Exact parent derivation and the current publisher documentation are being checked separately. A freshly fold-excluded image teacher can therefore test incremental image supervision on the weak outer split, with its existing model-search and report-label limitations. [Exact identity and source-card receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/initiative/audits/adapter-source-research-v1/sources.json).

The four endpoints give a descriptive interaction contrast:

`full - values-only - coverage-only + original = +.0554711 macro AUC`

This is arithmetic on completed endpoints, not an interaction significance test. AUC is nonlinear, targets are extremely sparse, training is stochastic, and masks change effective loss weighting. The CPU-accepted receipts verify the two recipe factors and matching seed/schedule/configuration. Use the existing paired outputs for an aggregate-only interaction analysis if authorized in the owning environment. Do not rerun four fits merely to obtain this arithmetic.

## Soft blending is not new loss mathematics

For a common logit and nonnegative weights:

`a BCE(y,p) + b BCE(q,p) = (a+b) BCE((ay+bq)/(a+b), p)`

Thus a separate teacher BCE is equivalent to a blended target when weights and normalization match. The substantive choices are label source, membership, masks, which image evidence is visible, and total gradient weight. An extra loss term can otherwise appear helpful simply because it changes effective weighting.

### Minimal proposed supervision contrast

- Control: the exact accepted baseline, with compatible existing control reused only after full source/input/initialization/optimizer equivalence
- Candidate: same architecture, views, initialization, masks, schedule and checkpoint rule; replace a prespecified fraction of target weight with independently produced image-teacher information, preserving total weight
- Diagnostic control: a predeclared conditional-mean or within-target/label/missingness-stratum shuffled-teacher version, to test whether image-specific ordering matters beyond generic softening. This is a diagnostic, not a replacement for a full matched control
- Initial scope: the ORIGINAL accepted-v2 observed mask (13,760 of 26,964 fitting cells), if the question is correction versus softening; the current full-public overlay treats all 26,964 cells as observed. Freeze the actual baseline values: adding image information to the current public-overlay recipe is a different contrast from blending it with accepted-v2 binary labels. Adding previously masked cells is a separate factor. Do not silently combine mechanisms or switch baseline values
- Go: teacher ancestry excludes outer components, input is compatible, useful teacher ordering is measurable, runtime fits the remaining schedule, and a matched fixed-final result supports advancement with all 12 targets reported
- Stop: unresolved eligibility, an impossible runtime, numerical failure, no retained image-specific teacher information, or improvement entirely explained by altered exposure/weighting. An uncertain small pilot is not proof of equivalence

A clean nested two-fold teacher construction inside the 2,247-study fitting set requires two half-fit teachers, each predicting the opposite half. Neither sees any of the 1,156 outer studies. Two fresh students then compare baseline and blend. Under equal epochs and equal per-image cost, this is approximately three baseline-equivalent fitting exposures, plus inference; it is not a cheap two-model experiment. Using the observed 18.36-hour baseline as a rough same-hardware anchor gives about 55 GPU-hours, plus inference; keeping a full 2,810-update budget for each half-fit teacher raises the estimate toward 73 hours. These are planning estimates, not measured runtimes. Half-sized teachers may also be weaker; a compute-saving split does not guarantee useful pseudo-labels. A teacher trained on all fitting studies is outer-excluded but produces in-sample fitting labels, which is a different hypothesis.

Fresh cross-fitting does not erase upstream gold-guided design or pretrained teacher exposure. Distinguish those cases: the current label-source receipt documents report-LLM derivation and gold-guided choices, not outer-image teacher exposure. In contrast, the imported .949 state was trained on all 4,349 nongold studies, encompassing our internal outer and reserved confirmation cohorts. Using that state changes the validation contract even if later fine-tuning is fitting-only. Report exactly which layer of independence has been established.

## Resolve existing work before adding mechanisms

1. Obtain the completed second-seed supervision result when lawful access returns. Its training completion does not establish an evaluated result
2. Recover or explicitly close the dense-adaptation comparison using its actual cancellation lineage. Signal-15 cancellation is not intentional checkpoint exit75
3. Complete the already-designed whole-bag mean versus query comparison. Both arms must see the same frames and receive the same study-level supervision. Comparing a four-window bag with a one-window fit changes exposure roughly fourfold
4. Partition the 501 all-masked training-split rows excluded before overlay. Some may support admitted image-only learning; do not admit all rows or move protected confirmation/gold/quarantine boundaries
5. Consider the untested same-budget complementary-series selector only after descriptor semantics are checked. Existing evidence found omitted available non-fluid categories in 214/2,247 fitting studies. The bundled source-slot failure does not isolate this question

## Deadline and deployment gate

[Official competition overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/): entry/team merger October 15, final submission October 22, winner deliverables November 5, all at 23:59 UTC. A three-example notebook runtime is not evidence of full hidden-test compliance. Every promoted recipe needs actual hardware-matched full-scale inference, offline dependencies, exact source/weight binding, and margin inside the nine-hour limit.

The goal is a strong, eligible, reproducible solo model. None of the evidence supports a promised winning score.
