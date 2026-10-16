# Actual late-epoch results and compact challenger startup

The owner identified research commit cf3819cb0a87c282291345929b5266356f067a70 on main. It has been incorporated. The N3 whole-recipe compact challenger is the next priority; more LoRA averaging is deprioritized.

## Actual competition evidence

Authenticated Kaggle outcomes were reconciled once into Newton's canonical ledger: E8 submission 57221755 = **0.863**, E10 submission 57245976 = **0.864**, both scored. E5/E6 remain the best trajectory checkpoints at **0.872**. Later epochs did not improve this public sample; this alone does not establish overfitting or private ranking. Neither entry may be resubmitted. The reconciliation performed no Kaggle mutations.

## New compact challenger

Native-CoAtNet-Bag12 N3 uses a fully trained ImageNet-initialized CoAtNet trunk, within-series three-slice windows and disease-specific study attention. The dossier specifies fixed16 epochs on2247 fitting studies and final-only1156 outer evaluation with the existing soft fitting overlay; no gold/confirmation tuning. This is a whole-recipe comparison, not causal backbone attribution.

Official initialization revision2c31d82376cc244c907c064332941e6c65be81f1 and timm1.0.30 were acquired and verified. Strict original classifier-state loading had zero missing/unexpected keys; removing its classifier leaves72,856,264 trainable trunk parameters/1024 features/25 BN modules. All16 positive hidden encoder dropouts were found and disabled. Actual model inventory is CPU engineering only, not MRI accuracy.

Initialization and runtime are staged privately on evc40 at /tmp/rsna-compact-native-n3-assets-20261016-v1. CPU929905 completed; existing native cache/runtime remain intact. No compact GPU training has been dispatched. Executable model/sampler/resumable trainer/inference implementation is assigned to [Devin issue91](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/issues/91). Root owns actual GPU acceptance and dispatch.

## Concrete older recovery failure and correction

Instrumented original919305 reference child returned0/four finite updates. Control child returned1 with Triton driver C compilation failing because Python.h was absent. This explains that recorded control failure, not all historical failures. Matching Rocky Python3.12.13-3.el8_10 signed development RPM was verified and extracted privately; system runtime pyconfig-64.h completes those headers. CPU929918 actually compiled the original Triton driver successfully. No global package/security configuration changed.

Distinct recovery929938 was submitted once to evc40, one H100/two CPUs/32GiB/one hour. At the last actual check it was PENDING Priority with zero runtime. It preserves frozen bf46b70 numerical source and inputs, adding only matching private headers and node-private compiler caches. It is engineering recovery, not a production fit or accuracy result. Do not restart it, repeat staging or infer success before terminal evidence.

No clinical images/reports/weights are included in this milestone. Exact local stage and dispatch receipts are retained alongside this document.

## Root implementation and executable capacity probe

Root implemented `rsna/competitive/compact_bag.py`: exact same-series triplets, canonical SHA256/local CPU torch sampler, no-replacement two-per-series/fill coverage, one-time mean/std.5 normalization, strict pinned ImageNet initialization, zero encoder dropout and full-dimensional per-target attention. Evaluation chunks embeddings with saved BatchNorm before one global bag softmax; training chunking is refused. The new module actually loaded the original weights strictly:72,856,264 encoder/73,136,096 total parameters, all trainable, only MIL dropout. No clinical forward was performed in that CPU check.

Eighty focused/adjacent tests passed, including small numerical full-versus-restarted model/BN/AdamW/RNG replay, channel/bounds/masks, missing and short series, exact sampler, gradient flow and eval chunk consistency. Ruff and scoped mypy passed. These tests are not full clinical trainer recovery or an accuracy result. Devin issue91 now explicitly owns the trainer/inference integration while root owns this module, avoiding duplicated implementation.

Distinct synthetic capacity job930089 was submitted ONCE, one H100/two CPUs/32GiB/15min on evc40. It tests twelve-window and one-window BF16 forward/backward, finite gradients, changed trunk and single BN update; no clinical data/checkpoint/output prediction claim. It remains PENDING Priority at the actual latest check. Existing assets are reused, new outputs/caches bounded128MiB node-private, shared growth zero. Both evc40 GPUs are currently allocated to unrelated users; scheduler forecast was Oct16 20:19 cluster time for this capacity job, not a guaranteed start. Do not repeat either queued recovery or capacity job.
