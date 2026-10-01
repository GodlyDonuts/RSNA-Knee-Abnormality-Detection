# Creative solo DINOv3 research, October 1

The owner requests independent research, including inspiration from natural
systems and other fields. This is now a standing instruction in AGENTS.md.
Creativity should produce discriminating experiments. Winning performance is a
goal, not an observed result, and novelty alone is not an admission criterion.

## Working diagnosis

Our MRI-only encoder adapters can learn from the accepted report supervision,
but sparse observations, incomplete labels and scanner/protocol differences may
limit generalization. None of these is a confirmed explanation of the gap
between internal AUC and Kaggle. A larger rank or more epochs will not isolate
these explanations. Keep the accepted labels/masks/folds and the official base
weights fixed while testing one mechanism at a time.

The full matched slice-coverage comparison is now training on two H100 nodes:
stratified866716 on evc32 and the distinct uniform recovery866719 on evc31. Both
have actual finite optimizer progress. The original uniform866715 failed device
preflight before any update. Native384 rank32 remains separately staged on frozen
C9 source, awaiting original-source jobs866426/866427 and assembly866458. No third
GPU job is admitted. Read actual receipts rather than restarting these jobs.

## First: change observation coverage at the same budget

An analogy to repeated physical measurements suggests taking different samples
on repeated visits rather than observing the same locations indefinitely. The
implemented stratified policy chooses one slice per ordered bin each training
epoch. It preserves series selection and deterministic evaluation. It does not
assign an examination-level abnormality to each slice.

The exact training-metadata replay gives 185,929 distinct slices for the current
fixed policy and 364,647 for stratification over ten epochs, with identical
1,859,290 slice presentations. This changes selected-series exposure from 48.06%
to 94.26%; it is not evidence of a prediction improvement. The fixed six-series
cap still omits 837 training series.

The [paired protocol](../initiative/experiments/competitive-v1/stratified-slices-v1/README.md)
has control and candidate configs differing only in slice_sampling. Actual CPU
cache decoding, interrupted training and offline inference passed. The pinned
official-weight H100 pilot866624 passed exact decoded recovery, frozen-base
invariance and equal initial states/logits; its26 private objects are fully
download/hash verified. Full-cache CPU acceptance866694 also passed. See the
[exact acceptance audit](../initiative/audits/stratified-slices-pilot-launch-v1/README.md)
and [actual production launch](../initiative/audits/stratified-slices-production-launch-v1/README.md).
Both full production arms repeat initialization acceptance with equal learned
initial states and update budgets. Reject the hypothesis if held-out
ranking fails to improve under the predetermined comparison. A tiny pilot or a
comparison to an unmatched old run cannot answer it.

## Second: learn relationships between observations

Different MRI planes and pulse sequences observe the same three-dimensional
structures. Inspired by multi-sensor estimation, test an auxiliary objective
that teaches DINOv3 patch features about physical relationships, without using a
second teacher or changing the pretrained base weights.

Start within a single series: use verified image geometry to predict relative
slice displacement or correspondences between nearby overlapping regions.
Do not force every neighboring feature to become identical; a small lesion can
appear on only one slice. Compare with a control receiving exactly the same
images and classification updates without the relation objective. A shuffled
geometry control tests whether the benefit depends on spatial information or
only an extra loss. Any method that shrinks feature variance or erases localized
pathology evidence fails its engineering acceptance.

Cross-plane supervision is a later, separate test. DICOM positions/orientations
provide physical coordinates, but motion, oblique acquisitions, spacing and
field-of-view differences mean equal coordinates are not automatically matching
tissue. Establish registration and uncertainty first; reject pairs without
reliable overlap. Compare geometry-guided correspondences with same-examination
global consistency and shuffled-coordinate controls. Preserve contrast-specific
features rather than forcing T1, T2 and PD appearances to be identical. Without
verified correspondences, do not claim cross-view anatomy supervision.

This objective can use admitted unlabeled knee examinations. It therefore does
not require fabricating twelve disease labels for a source with only ACL and
generic meniscal annotations. Train auxiliary heads only on the training cohort;
discard them for the final MRI-only classification interface. This proposal is
now has an isolated [unsigned-distance prototype](../initiative/audits/slice-relations-prototype-v1/README.md).
It operates on raw global slice embeddings, not patch correspondences. It is not
integrated into production, trained on real MRI or clinically measured yet.

## Third: allocate detail where it adds information

Inspired by foveated vision and adaptive measurement, test a two-pass solo model:
first inspect the full examination at a broad scale; then use the same encoder
and learned target queries to request a bounded number of detailed regions.
Keep an exploration fraction to avoid a feedback loop that repeatedly ignores
subtle lesions. The model must aggregate evidence at examination level; an
unselected slice is not a negative example.

Compare at equal total encoded pixels/tokens, not equal crop count. Controls are
uniform high resolution and random detail selection with the same budget.
Evaluate per-target ranking, rare-positive sensitivity and stability to missing
series, with no gold/confirmation-based tuning. The selection could learn
scanner markers or acquisition artifacts rather than anatomy; this is a failure
mode to test, not an explanation to assume. Selection visualization alone does
not establish localization accuracy. Offline Kaggle memory/time acceptance is a
separate prerequisite. This is a proposal after resolution and coverage results,
not a replacement for the current production payload.

## External data: more independent patients and more useful supervision

The [source inventory](../initiative/audits/external-knee-data-inventory-20261001-v1/README.md)
records candidates and actual public fastMRI+ metadata inspection. No external
MRI image cohort is admitted, downloaded or included in training.

Primary source pages rechecked on October 1:

| Source | What it can contribute | Limitation relevant to our hypothesis |
| --- | --- | --- |
| [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/) | Public training 1,130 exams; multisequence ACL, generic meniscus and abnormality supervision | The 1,370 total includes 120 tuning and 120 hidden exams. Generic meniscus labels cannot supervise both compartments as positive. |
| [fastMRI](https://fastmri.med.nyu.edu/) and [fastMRI+](https://github.com/microsoft/fastmri-plus) | Broader acquisition diversity and spatial pathology supervision | Raw/DICOM/annotation counts overlap. Current image agreement restricts use, redistribution and dataset linkage; do not assume Kaggle or overlap-matching permission. |
| [OAI](https://www.niams.nih.gov/grants-funding/funded-research/osteoarthritis-initiative) | 4,796 participants with repeated imaging; possible unlabeled anatomy adaptation | Longitudinal scans are not independent patients or automatically twelve compatible disease labels. Keep related visits/knees grouped. |
| [SKM-TEA](https://aimi.stanford.edu/datasets/skm-tea-knee-mri) | 155 scans with tissue segmentations and 16 pathology boxes | Only 86 are its training split; qDESS differs from routine RSNA acquisitions. Source-specific rights and independence remain unverified. |

First external experiment, after admission: MRNet training-only auxiliary heads
with a source-specific label mapping, followed by RSNA fine-tuning. Compare with
an RSNA-only model receiving the same extra update and compute budget. Also
compare a labels-masked external adaptation arm if resources allow, to distinguish
anatomical exposure from label benefit. These arms are separate experiments;
none is dispatched. Do not merge a source's tuning/hidden partition by default.

Admission requires accessible source terms compatible with competition and
weight publication, a current host-rules check, exact version/membership,
permitted independence checks, label mapping and funded storage. Unannotated
targets stay masked. Hugging Face mirrors of RSNA data add no new patients;
mirror licenses do not supersede original source restrictions. KneeCoT remains
excluded under the preserved host ruling. The rules page returned no readable
body on this web check, so fresh organizer permission is not claimed.

The later signed-in Stanford review confirmed MRNet data access, but its accepted
research agreement restricts derivative works and redistribution. Public
competition-weight compatibility remains unresolved. The imported label tables
also contain one fewer record per split, with first-case-like column names.
Resolve original CSV parsing and source rights before any MRNet training
admission. See the [specific receipt](../initiative/audits/external-knee-data-inventory-20261001-v1/mrnet-redivis-access-review.json).
The owner accepted leaving KneeCoT unused after describing a Discord update;
do not repeatedly ask for that clarification or treat it as a verified reversal.

A recent [KneePreM preprint](https://arxiv.org/abs/2609.31461v1) reports self-supervised
pretraining of a 3D U-Net on 19,011 OAI series from 4,791 participants, followed by
transfer to several knee tasks. This is relevant prior work, not a demonstration
that our DINOv3 relationship objective will improve RSNA. Its different tasks,
architecture and evaluation prevent comparing its AUC directly with our Kaggle
score. The abstract was inspected; full methods/code/weights and image rights
have not been accepted. Do not present generic unlabeled knee adaptation as our
novel discovery or use that preprint's APM data as an accessible cohort without
further source investigation.

## Decisions and stopping rules

The standing directive includes ideas that appear contradictory. Treat the
contradiction as a question to resolve: broader slice exposure might discover
small lesions, but could also dilute evidence with uninformative anatomy;
spatial consistency might stabilize features, but could erase a focal lesion;
adaptive detail could sharpen predictions, but might repeatedly overlook an
unfamiliar finding. The paired controls above test these competing mechanisms.
Keep the observation budget explicit and use examination-level supervision.
Confidence in an analogy is not evidence that its mechanism works.

Record each experiment's mechanism, expected per-target benefit, control,
parameter/token/update budget and falsifier before launch. Report all twelve
targets and patient/leakage-group uncertainty, including undefined rare-target
bootstrap draws. Use the fixed final epoch for the paired development comparison;
repeat the promising result on another seed/fold before a broad claim. Protected
confirmation and gold remain unused for iterative selection.

Track whether a gain comes from additional patients, labels, spatial constraints
or additional compute. Preserve negative findings and exact input/checkpoint
provenance. Only promote a model after actual held-out benefit and reliable
MRI-only offline deployment. Stay within two GPU nodes/minimum CPUs, private
storage reserves and authoritative daily submission allowance. Dataset inventory,
software tests and publication receipts do not establish competitive accuracy.

## October1 scheduling based on actual acceptance

The full256 source/cache and stratified recovery pilot have now passed their
actual CPU/H100 gates, and the paired uniform/stratified ten-epoch arms866715/
866716 were submitted once. This directly tests broader temporal slice exposure
while native384 source reconstruction continues on CPU; it does not change the
frozen native scientific recipe. The existing two-node limit controls their
scheduling. Exact production audit is in
`initiative/audits/stratified-slices-production-launch-v1/README.md`.
New anatomy/physics objectives remain proposals until implemented, tested and
matched against controls. No new held-out improvement or novelty is claimed.
