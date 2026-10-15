# MeniMV and KneeDINO mechanism review

14 October 2026. Primary papers, official publication pages and author repositories were checked. No MRI data or model weights were downloaded, accounts created, agreements accepted or experiments run.

## Decision

**Neither warrants a new priority training branch before the competition deadline.** MeniMV is useful chiefly for recognizing the cost of anatomical and severity annotation. KneeDINO supplies a more concrete fusion hypothesis, but its evaluation has unresolved reporting problems. Reuse ideas only within existing matched experiments; neither is a ready severity model or an admitted weight source.

## MeniMV

**Evidence.** Methods describes 750 studies from three centers, yielding 1,500 expert-selected sagittal–coronal pairs and 3,000 images. The abstract instead claims 3,000 examinations and 6,000 images. Clinicians select informative anterior/posterior-horn views and exclude motion-degraded images. Stoller 0–III grading is named, but detailed operational criteria and an RSNA medial/lateral-meniscus mapping are absent. The model uses shared-view encoding, concatenation, separate horn classifiers, focal loss and feature-MSE alignment. This is not an implemented ordinal objective.

Leave-one-center-out results are reported, but ordinary train/validation/test sizes, patient-grouped splitting, center sizes and independent external-cohort evaluation are not documented. Same-patient image pairs make grouping important; the omission does not prove leakage. No isolated MSE-loss ablation establishes its contribution. [Primary paper](https://arxiv.org/html/2512.18437v1)

**Release.** The paper claims a release but supplies no implementation, weights or usable dataset-license link. No matching author implementation was located. The first author's related MeniOmni repository currently contains README/LICENSE and promises a later release; it is not a MeniMV implementation. Its MIT declaration cannot be transferred to MeniMV artifacts. The arXiv publication license concerns the article. [Author repository](https://github.com/ShuruiXu/MeniOmni) · [Publication record](https://arxiv.org/abs/2512.18437)

**Reuse judgment.** No-go for importing the recipe. Expert-selected horn pairs supply localization unavailable to our whole-study model. Separate anatomical heads largely overlap existing target-specific pooling; the new information would be reliable horn/grade annotations. Full-feature cross-view MSE could suppress complementary information, so it requires its own aligned-versus-shuffled-pair control rather than automatic adoption. Do not translate grade numbers into competition positives without a verified rubric.

## KneeDINO

**Evidence.** The paper uses three independent RadioDINO ViT-B/16 encoders, approximately 258M backbone parameters, with roughly 45M encoder parameters fine-tuned. Task-specific slice pooling produces three plane tokens per pathology; a shared two-layer, six-head transformer mixes them after plane-ID embeddings. It predicts three binary MRNet outcomes, not injury severity or physical extent. Fusion-versus-concatenation ablations report AUC gains of .022 ACL, .057 meniscus and .024 abnormality, without paired uncertainty or seed replication. Only the public 120-exam validation split is evaluated. Early stopping is stated, but its monitored split and any separate selection holdout are unspecified. External validation is future work. [Primary paper](https://papers.miccai.org/miccai-2026-sat/paper/EMERGE_021.pdf)

**Reporting issue.** The paper's stated validation prevalences imply about 23 ACL-positive and 40 meniscus-positive cases. Stanford says each standard 120-exam subset has at least 50 positives per label. Its stated training abnormal prevalence also cannot reconcile with Stanford's full-cohort total. The actual split/reporting discrepancy needs resolution; it is not proof of patient leakage. Stanford documents patient-disjoint splits of 1,130/120/120 exams. [Official MRNet split and cohort](https://stanfordmlgroup.github.io/competitions/mrnet/)

The original MRNet headline results used as a comparator come from its internal-validation cohort, now the hidden test set; KneeDINO reports the public/original-tuning subset. Thus those headline numbers are not a same-split comparison. [Original MRNet paper](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699)

**Release.** No KneeDINO code, fine-tuned weights or corresponding license was located; the official page lists no supplement. Upstream RadioDINO is separate: its repository LICENSE says MIT, its README says CC BY 4.0, and its model card declares CC BY 4.0. This discrepancy needs clarification before reuse and does not license unreleased KneeDINO weights. [Publication page](https://papers.miccai.org/miccai-2026-sat/EMERGE_021.html) · [RadioDINO license](https://github.com/Snarci/Radio-DINO/blob/main/LICENSE) · [Repository](https://github.com/Snarci/Radio-DINO) · [Model card](https://huggingface.co/Snarcy/RadioDino-b16/blob/main/README.md)

## What is actually new relative to our model

Existing native attention already pools per target; existing PlaneMemory pools each sequence and lets a target query attend across sequence summaries. KneeDINO's more specific difference is **two rounds of self-attention updating the plane tokens themselves**, rather than one query-to-sequence readout. Neither provides anatomical registration. [Our plane-memory implementation](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/8300cfd20c2acaeef9d684a81dbc7fe977282a62/rsna/competitive/plane_memory.py)

That interaction operator is conditionally worth testing only as a later extension of the existing plane-memory lineage: retain one shared encoder, identical plane summaries, plane identities, all twelve labels, parameter-matched readout, fixed exposure and endpoint. Compare the current query readout with the small token-update mixer. Do not simultaneously copy three encoders, add external pretraining or change supervision. Retain an unchanged baseline and a diagnostic that shuffles image evidence relative to fixed plane identities; jointly permuting whole tokens and their identities would merely test expected set invariance. The paper supplies a hypothesis, not an expected RSNA gain.

MeniMV does not make missing severity concepts identifiable, and KneeDINO offers no severity mechanism. Neither supports treating attention weights as lesion localization or Gemini-generated descriptions as validated clinical explanations. Completing the existing controlled work remains the better near-term choice.
