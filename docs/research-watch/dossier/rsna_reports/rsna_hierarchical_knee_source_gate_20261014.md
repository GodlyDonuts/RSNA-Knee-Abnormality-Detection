# New hierarchical knee evidence: keep the severity branch deferred

14 October 2026 · Primary-source and released-code audit

## Decision

Neither source earns a new final-week severity experiment. The useful new finding is a **reproducibility falsifier in the coarse-to-fine implementation**, plus a **source/access no-go for KJD-25K**. Neither supplies a validated mapping to the three RSNA OA compartments or independent truth that repairs the report-supervision mismatch. Retain the existing compact-challenger / conditional teacher-blend priority.

## 1. Coarse-to-fine OA/KL: what was actually measured

The current paper is v3, dated 19 August 2026. It studies MRI prediction of five KL classes and binary OA defined by KL ≥2. Thus the two labels are deterministically related; the binary label adds no independent annotation. The KL head uses ordinary multiclass cross-entropy, not an ordinal loss. Single-KL OA evaluation sums probabilities for grades 2–4; it does not rank mean severity. This does not challenge the earlier expected-severity versus threshold-risk counterexample. [Paper v3](https://arxiv.org/html/2605.00718v3), [evaluation protocol](https://github.com/jukieCheung/coarse2fine-oa-mri/blob/9059db990bc3c636f9373db7fd617cb9390f9a45/stats_recomputed_protocol.json)

Reported aggregate results, independently readable without patient predictions:

- M3T OA AUC: 0.792075 → 0.917483; difference 0.125408, paired-bootstrap 95% CI [0.048387, 0.207601], BH q=0.0036
- ResNet3D OA AUC: 0.880653 → 0.865734; difference −0.014918, CI [−0.069812, 0.040762]
- nnMamba OA AUC: 0.850350 → 0.819114; difference −0.031235, CI [−0.114712, 0.050092]
- ResNet3D KL macro-AUC: 0.672506 → 0.794386; q=0.006299. Its KL accuracy and macro-F1 also improve after correction

These are one OA endpoint and five-class KL metrics, not twelve-target RSNA macro-AUC. [Author aggregate statistics](https://github.com/jukieCheung/coarse2fine-oa-mri/blob/9059db990bc3c636f9373db7fd617cb9390f9a45/stats_recomputed_with_fdr.csv)

The paper declares 383 training / 98 test subjects, one examination each, final-epoch evaluation, and training-only PCA/probe fitting. However, every configuration was trained once without explicit random-seed control. Subject bootstrapping does not estimate training-seed variability, and there is no external validation. [Paper, §§3.2–3.4, 5](https://arxiv.org/html/2605.00718v3)

### Concrete falsifier: the released model is different

At commit `9059db990bc3c636f9373db7fd617cb9390f9a45`, both training and evaluation implement:

    MRI → backbone → five KL logits → Linear(5,2) → OA probability

That is a five-logit bottleneck, whereas the paper describes two parallel heads on shared encoder features. Two-class CE versus BCE is not itself the decisive discrepancy: CE equals BCE on the logit difference. The feature-level versus KL-logit-level connection is substantive. Therefore the public implementation falsifies the assertion that the described parallel-head experiment is presently reproducible as written; it does not prove the reported aggregate numbers wrong. [Training implementation](https://github.com/jukieCheung/coarse2fine-oa-mri/blob/9059db990bc3c636f9373db7fd617cb9390f9a45/train_oaizib_3models_dual.py), [evaluation implementation](https://github.com/jukieCheung/coarse2fine-oa-mri/blob/9059db990bc3c636f9373db7fd617cb9390f9a45/eval_oaizib_aclr.py)

## 2. Modality and anatomy do not match the RSNA targets

The upstream CartiMorph paper explains the 481-case subset: KL grades could be retrieved from OAI for 481 of 507 right-knee DESS scans; its exact stratified split and class counts match 383/98. This supports the subject-level split account, although this audit did not inspect patient rosters. CartiMorph also explicitly excludes patellar segmentation. [Upstream data provenance, §4.1 and §5.4](https://arxiv.org/html/2308.01981v3)

OAI documents KL as an X-ray assessment, with MRI cartilage/morphology and MOAKS assessments separately defined. Neither new paper documents independent MRI-derived, compartment-specific KL labels. Predicting radiographic KL from MRI remains an MRI-input task, but it is a different reference target. Exact OAI variable/version/date linkage in coarse-to-fine remains unspecified. [OAI assessment documentation, §§3.1–3.5](https://nda.nih.gov/static/docs/ImageAssessmentDataOverview.pdf)

RSNA requires separate medial tibiofemoral OA, lateral tibiofemoral OA and patellofemoral OA probabilities among twelve findings. A single KL ≥2 indicator cannot be assigned to all three without unsupported compartment assumptions. [Official RSNA target definitions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)

Anatomy masks exist upstream, but the coarse-to-fine training loader uses only images and KL grades; masks are a post-hoc saliency check, not anatomy supervision. Its default overlap ROI uses labels 2 and 4. The advertised OAIZIB-CM release instead defines 2=femoral cartilage, 4=medial tibial cartilage and 5=lateral tibial cartilage. Thus using CM masks unchanged would omit lateral tibial cartilage. Original OAI-ZIB masks could explain the choice; the actual mask version/remapping must be established before interpreting the overlap. [Overlap code](https://github.com/jukieCheung/coarse2fine-oa-mri/blob/9059db990bc3c636f9373db7fd617cb9390f9a45/compute_saliency_overlap_metrics.py), [official CM label map](https://huggingface.co/datasets/YongchengYAO/OAIZIB-CM/blob/main/README.md)

## 3. KJD-25K: useful future resource, unavailable evidence for this decision

Its labels originate in Chinese reports, with Qwen3-Max candidate extraction and clinician review. Independent image-only severity rereading is not documented. It reports a single-hospital 20K/5K example split without an explicit patient, bilateral-knee or repeat-visit separation guarantee. Acute trauma within 72 hours is excluded, creating another population mismatch for injury targets.

The measured endpoint is report generation: the trained model reports BLEU 42.71, F1 3.26, and GPT-4o-assigned LS/DS/SRC scores 79.63/58.47/66.42 on 0–100 scales. These are not binary AUCs or separately validated MRI severity measurements. No severity-head ablation isolates the contribution of graded labels. Anatomy/severity fields are report-derived text; a dense-mask or RSNA-compatible compartment-severity release was not established.

Training uses Qwen3-VL LoRA for three epochs on H100 hardware; wall-clock/GPU-hours are unreported. The curation section’s 2,000 clinician-hours plus 2,000 GPU-hours must not be mistaken for model training time. [Accepted paper, §§2–3](https://papers.miccai.org/miccai-2026/paper/0576_paper.pdf)

The MICCAI page lists code as N/A. Author feedback names Qwen3-VL-Instruct-8B and promises controlled-access CC BY-NC-SA 4.0. The linked KneeCoT card instead displays CC BY-NC 4.0, a contact-sharing gate, and a mandatory signed hospital ethical-use agreement; it lists 1.05 TB. Its broader 600K VQA / 50K reasoning claims and different title also mean the linked release should not be assumed identical to the accepted experiment. No approved checkpoint or license-to-use package was verified. These are admission questions, not permission to accept terms or acquire data. [MICCAI release and author feedback](https://papers.miccai.org/miccai-2026/0534-Paper0576.html), [linked release metadata and terms](https://huggingface.co/datasets/YiHui0124/KneeCoT)

## 4. Readiness and stopping condition

Coarse-to-fine has readable source and aggregate statistics, but no LICENSE file, checkpoint files or GitHub releases in the inspected public repository. GitHub metadata reports no detected license. The paper’s CC BY license is not a verified code/weight license. Its recipe is 100 epochs, batch size 2: 38,300 volume presentations and 19,200 optimizer batches per 383-case fit, calculated from the released loop. Hardware throughput and end-to-end RSNA deployment cost are unknown. Source imports also expect model modules at the import root although files are under `models/`, so the README command needs environment/path reconciliation. [Pinned repository](https://github.com/jukieCheung/coarse2fine-oa-mri/tree/9059db990bc3c636f9373db7fd617cb9390f9a45), [training recipe](https://github.com/jukieCheung/coarse2fine-oa-mri/blob/9059db990bc3c636f9373db7fd617cb9390f9a45/README.md)

**Stop here for the final-week branch.** Reopening would require a reproducible admitted artifact, matching head topology, explicit target/compartment mapping, patient-independent provenance, and a credible resource envelope. None is supplied by these sources. The mask-label contradiction is a cheap metadata falsifier to resolve if this work is revisited; no new patient analysis is needed to establish that the currently documented contracts disagree.

Scope: reviewed public full text, source code, aggregate statistics and release metadata. No dataset files, per-subject prediction CSVs, model weights or RSNA clinical rows were opened. No training, inference, repository edits, agreements, outreach or publication was performed.
