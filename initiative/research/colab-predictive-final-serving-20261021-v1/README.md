# Colab actual-final B/C serving diagnostic

The owner authorized up to the existing 300 Colab compute units. The separate full-data Colab training route stopped at HF storage capacity and its idle A100 was released; no training occurred there. This diagnostic uses a single T4, the already published immutable B/C finals, the already uploaded parent asset archive, and only the competition’s three public MRI examples. It requires no additional HF uploads, training-cache transfer, Kaggle GPU allocation, or competition entry.

Measure exact trained FP16/FP32 probabilities, repeated inference, CPU preparation time, GPU/host memory, and C’s response to replacing regional descriptors with within-study means. Read all three complete public studies. Reuse the original renderer, numerical modules, parent and five-object publication proofs. Credentials enter masked runtime prompts and remain in RAM; no credentials in notebook source.

This is a Colab single-T4 runtime diagnostic, not canonical offline two-T4 Kaggle qualification, clinical accuracy, an independent validation set, or a new model. It cannot spend the protected primary Kaggle allocation. Release the runtime once aggregate results are collected. The existing original Newton fitting/deployment/preservation owners remain unchanged.

Before execution, source-envelope path/ZIP checks and compilation passed locally. Actual execution results must be recorded separately.

## Actual input rate limit and retained-study decision

The first pass downloaded most of the557 public files, then stopped at an unrecorded non-success HTTP response. Its bounded continuation observed **HTTP429**. All originals were retained. The next diagnostic uses only the **two complete studies /419 DICOMs /527,406,648 bytes** already present, verified against original API membership and sizes. It requests no more DICOM files; the third incomplete study is left intact and excluded. This availability-only selection uses no labels or predictions. It explicitly cannot replace the three-study competition qualification. Exact source is767499a400bdb184596f1208bde987a6ad1617fd.

## Actual result — October 21, 10:15 UTC

Both exact fixed-final models completed inference on the two intact public studies. Each used48 windows; the cases contained284 unsigned12-bit frames and135 signed16-bit frames. B warmedFP16 times were0.544/0.543seconds; C0.551/0.550seconds. Largest FP16-versus-FP32 probability differences were0.000429 forB and0.000147 forC; repeatedFP16 outputs were identical. Peak allocated GPU memory was1.363/1.364GB and process peak hostRSS3.741GB. CPU decode-plus-render times were25.495 and6.021seconds, dominating inference. These are ColabTorch2.11/timm1.0.29/single-T4 measurements with one CPU numerical thread, not hidden-scale timing or the original Kaggle image.

Replacing C’s regional target descriptors with within-study means changed a final probability by at most0.0176155 and0.000309676 on the two cases (1.76155/0.03097 percentage points). This demonstrates nonzero sensitivity to regional descriptor variation in these examples; it does not establish that the changes are correct, an accuracy benefit, or a controlled change from the earlier different fitting examples. No labels, reports, source-derived weights or training updates were used. [Full sanitized measurement](actual-result.json).

The completed aggregate was collected before releasing the T4. All source checkpoints remain in their original private publications; Colab produced no new trained weight. The primary three-arm, three-public-study, two-T4 offline qualification remains unchanged and still requires C0’s actual final. No Kaggle competition entry or GPU allocation was used by this experiment.
