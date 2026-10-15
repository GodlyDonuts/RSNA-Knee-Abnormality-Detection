# Late research findings and status on October 15 2026

This addendum preserves the later research screen and supersedes older status statements only where explicitly stated. No paper result below is a measured result on this competition model. No training, model inference, checkpoint download, or competition submission was performed for these research reviews.

## Current committed experiment status

Repository pin: [70c3a6fab25f47e208afd4f0388296bf3b2e948a](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/commit/70c3a6fab25f47e208afd4f0388296bf3b2e948a), committed October 15 at 22:40:14 UTC.

The [E10 portable public-start receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/70c3a6fab25f47e208afd4f0388296bf3b2e948a/initiative/audits/kaggle-epoch10-portable-20261015-v1/public-start.json) was observed at 22:37:21 UTC. Dataset 12613343 version 1 had all 68 hashes verified. Kernel 138929766, `godlydonuts/rsna-alldev-offline-e010-0f0ab3f260e9`, version 1, was running as a public example. `competition_submission_created` was false. This reproduces the original E10 payload source. It is neither a newly fitted model nor a scored competition result.

E8 is complete: public score .863, submission 57221755. Same-trajectory rounded public scores are E1 .752, E3 .856, E4 .865, E5 .872, E6 .872, and E8 .863. The best recorded result remains .872. E8's .009 decrease weakens indefinite epoch extension but does not establish overfitting or private-set performance. See the [scored E8 receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/c195690dc03e131ed210953cf1e22fd631eb364f/initiative/audits/kaggle-epoch8-direct-submit-20261015-v1/scored.json).

Retain the bounded E10 reference when completed. Compact N3 remains the higher-priority new controlled experiment; no new method in this addendum displaces it. A shared deployment Dataset discussed in issue 90 could avoid roughly 1.2 GB of repeated packaging, but that is a deployment improvement, not accuracy evidence.

## Official rule and evidence caveats

The October 15 indexed official competition screen retained the final deadline of October 22, 2026 at 23:59 UTC and offline CPU/GPU submission runtime of nine hours. Direct Kaggle page bodies were empty, so this was not an exhaustive same-day forum audit. The nine-hour submission limit must not automatically be treated as a training limit for training outside Kaggle. Sources: [official competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/) and [RSNA announcement](https://www.rsna.org/media/press/2026/2669).

A newer-looking Versia .956 claim used multiple weight collections, so it does not strengthen the evidence for a single MRI-only model. Public leaderboard claims without a bound executable version, checkpoint count and provenance remain claims rather than reproduced measurements.

## A4-MR

[Primary paper, Table 4](https://arxiv.org/html/2610.13056v1#S4.T4): an abnormality-aware contrastive loss improved whole-volume liver AUROC from 96.1 to 98.9. Under the identical padded predicted-crop setting, the comparison was 99.8 to 99.8. The knee evaluation concerned segmentation rather than this competition's classification objective. No author implementation was located during the screen.

Interpretation: the whole-volume improvement cannot be transferred to a cropped knee classifier as an isolated loss benefit. The matched crop result is the relevant caution. No new experiment was promoted. A future comparison would need identical crop, labels, architecture and compute with only the auxiliary loss changed. Stop before implementation if its prerequisites or transferable supervision are unavailable; go only if an independently evaluated gain justifies the added training cost within the remaining schedule.

## RadiomicsPersona

[Primary paper](https://arxiv.org/html/2601.08604v1#S5.SS4) and [author implementation](https://github.com/YaxiiC/RadiomicsPersona). The January 2026 paper extends MICCAI 2025 work and is not independent confirmation of that earlier result.

Healthy diffusion persona plus the full radiomics route changed abnormality AUC from .85 to .76, ACL from .92 to .84, and meniscus from .82 to .87, with substantial variability. Registration, diffusion generation and radiomics extraction introduce meaningful computational and integration cost.

Interpretation: mixed target behavior and setup cost do not support a final-week replacement. No run was proposed. Any later test should retain the same imaging backbone and data boundary and isolate the extra persona/radiomics inputs, with per-target uncertainty and total extraction cost included. Stop if improvement is concentrated in one target while the primary aggregate degrades; go only on a prespecified primary improvement with feasible deployment.

## KAMRNet

[Primary paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC12508577/) and [author model](https://github.com/zqiuak/KAMRNet/blob/main/model_forallplane_MMAL.py). The route uses meniscus segmentation plus nine separate binary classifiers. Its three-plane ResNet18 system processes whole images and detached-attention crops, with thresholded components and 24 slices.

The screen did not verify an equal-compute crop control. Nine classifiers also do not directly meet the single-model objective. Anatomical focal evidence remains plausible, but this is not evidence that duplicating the route will outperform the present native model. Defer. A later matched control would allocate the same image-encoding budget to whole-image or focal views inside one model, keeping training labels and validation independent. Stop if localization supervision, compute parity, or independent evaluation cannot be established.

## KneePreM

[September 2026 paper](https://arxiv.org/pdf/2609.31461) and [author implementation location](https://github.com/BiomedSciAI/fuse-med-ml/tree/master/fuse_examples/imaging/oai_example). The matched pretraining comparisons concern knee MAE against random initialization and abdominal-CT SuPreM. Reported full macro results were .777 on fastMRI+ and .787 on APM.

No matched DINOv3 or modern 2.5D comparison was established, and no downloadable checkpoint was verified. This is a pretraining research lead, not a ready substitute. Stop if weights and eligibility cannot be verified without undertaking fresh pretraining. Any future go decision should compare initialization only with matched downstream data, architecture, budget and independent endpoint.

## Positive weighting and the current BCE denominator

The [nartaa .949 author discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/746792) mentions RSNA weights, without an isolated ablation. The [mathischmp implementation](https://github.com/mathischmp/challenge_rsna/blob/26876d2ab251d95e2b1d7611d0d979001e41b6e3/src/rsna_knee/window_train.py#L288-L299) derives soft-label prevalence, clips it to [.03, .70], and clips `(1-p)/p` positive weights to [1, 10]. Its .920 is evidence for a complete recipe, not an isolated loss gain.

For a target with constant positive weight a, the population BCE optimum is a E[q|x] / (1 - E[q|x] + a E[q|x]). This is monotone in E[q|x]. Ranking gains would therefore depend on finite-capacity optimization or other departures from that ideal, not a generic ranking advantage of the weighted optimum. An isolated q=.5 target has optimum 10/11 under a=10; this algebra is not a measured clinical harm.

[AqlanAhmed's report](https://github.com/AqlanAhmed/rsna-knee-mri/blob/9e3fe19b8467a96d7c4f1de725b1fb8f88c5c98d/README.md#what-did-not-help) claims MCL degradation of .115 on Gold58. Signed paired receipts were not verified; its `vision/train_raptor.ipynb` uses plain BCE. This is a caution, not a new controlled negative result for this repository.

The c195 repository loss review found that the global per-target effective-batch mean equals global BCE when all masks are one. No obvious denominator defect was established. Compact N3 retains plain BCE. No new positive-weighted, focal-loss or class-sampling run is promoted. If revisited later, isolate the loss on identical soft labels, sampling, seed, budget and split; require primary endpoint improvement with per-target uncertainty rather than promoting on a single rare-target fluctuation.

### Timing caveat from the public implementation

The mathischmp source `kaggle/train_window/run_train_window.py` and README describe 12 epochs in roughly seven hours on two T4 GPUs. Synthetic throughput was 26.7 windows/s; the reported actual rate was about 23 windows/s with 32-minute epochs. The schedule was shortened from 16 to 12 so scoring would fit in about 8.3 hours. It uses gradient checkpointing and four studies per GPU. Compact N3 uses no checkpointing and a one-study microbatch. These differences prevent inferring N3's fixed-16-epoch or all-development refit time, and do not justify automatically shortening an H100 run.

## OrthoFoundation

This is the strongest new pretraining contender in the screen, but it remains deferred.

Sources: [paper](https://arxiv.org/pdf/2601.18250), [pinned author README](https://github.com/ytrsk/OrthoFoundation/blob/4ae0a0aa1a5eedf578c1ec8db3e82bd9fac68470/README.md), and the [official DINOv3 forward implementation](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/dinov3/models/vision_transformer.py#L222-L234).

The pretraining collection contains 1,251,655 images: 357,670 radiographs and 893,985 MRI images. Downstream evaluations use full fine-tuning across different tasks. The June 18 release at commit 356d6cf78d68721c3f21133ba4a37d5fa303ba88 lists an LFS payload of 1,213,056,638 bytes with SHA256 385a775822107b68eaa486336feb982e1ce7bd6d4e8c03ceb482a0bf546f2ff9. The August 25 MIT license appears at commit 4ae0a0aa1a5eedf578c1ec8db3e82bd9fac68470. These are release metadata, not a downloaded or validated model.

### Compatibility uncertainties

- The paper's methods refer to DINOv2 while the results/README refer to DINOv3
- `exp/pretrain/dinov3/full/config.yaml` explicitly specifies ViT-L/16 with all 24 blocks
- `nets/dinov3_vit.py` loads `dinov3_vitl16`, but its wrapper drops the grid metadata from token preparation and calls blocks without RoPE, unlike the linked official forward path
- There is no exact release manifest binding the checkpoint to training code, base hash and teacher/student state
- README batch size 32 and cosine scheduling disagree with configuration batch size 24 and Polynomial scheduling
- `tools/utils.py` intersects keys and fills defaults before strict loading, so apparent strict-load success alone does not establish that every required weight was supplied
- The Meta Hub namespace differs from this project's Hugging Face namespace; existing LoRA adapters and task heads cannot simply be reused with a new base
- The projection head in `nets/multicrop_wrapper.py` is an SSL projection head, not the competition's twelve-disease classifier

### Eligibility uncertainties

Pretraining includes OAI, fastMRI and private clinical material. [The host OAI discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/741819) is conditional; [the public-checkpoint/private-pretraining question](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/737950) had no visible host answer in this screen. An MIT repository license does not itself resolve upstream dataset or base-model obligations. Consult [fastMRI terms](https://fastmri.med.nyu.edu/) and the [official DINOv3 license](https://github.com/facebookresearch/dinov3/blob/6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md). The evidence does not establish either blanket permission or blanket prohibition.

### Matched future comparison and stop/go

Change only the pretrained base, initialize fresh adapters and disease head, and hold inputs, labels, split and compute budget fixed. Go only after compatibility and eligibility are resolved and a completed independent gain leaves time for final refit and deployment. Stop otherwise. No implementation, weights download or model run was performed. Compact N3 remains ahead of this candidate because the release and eligibility uncertainty consumes the scarce final-week budget.
