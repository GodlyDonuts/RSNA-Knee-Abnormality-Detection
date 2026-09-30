# DINOv3 A32 fold-0 completed result

The fixed-final ten-epoch A32 model achieved **0.8620940794 internal macro AUC**
on the same **1,156 held-out development studies** where the original frozen-encoder
fold-0 baseline achieved **0.7783102649**: a **+0.0837838145** change. All twelve
target AUCs increased on this split. This is accepted report-derived supervision,
not clinical gold, the reserved confirmation set, a three-fold result or Kaggle.

Production Slurm **864856 COMPLETED 0:0** in **7:03:29**, with 2,810 optimizer
updates on 2,247 training studies, one H100/two CPUs and empty stderr. The
training source is `6f1ee6d332de29a8d6e5807aeac4777620865933`; original accepted
labels, masks, folds and pinned official DINOv3-L weights were retained. The
model has 10,426,636 trainable parameters, including 10,223,616 encoder adapters.
Input resolution remains 224 from the existing uint16 256-side cache; native
resolution and explicit neighboring-slice interaction have not been trained.

`recompute.py` independently joins both prediction files to the immutable
accepted labels, applies the exact target masks and reproduces every reported
AUC. It verifies matching training/held-out UID hashes, fold, release, cache and
source bindings. Run it with the repository QA Python. `comparison.json` also
preserves the changed recipe settings: encoder adaptation, precision, head
learning rate, warmup/cosine schedule, chunking and world size. The effective
batch remains eight. The matched BF16 frozen control is not complete, so this
comparison cannot isolate the adapter's causal contribution.

| Target | Frozen baseline | A32 adapter | Change | Positive / negative studies |
| --- | ---: | ---: | ---: | ---: |
| ACL | 0.7335 | 0.8846 | +0.1511 | 85 / 879 |
| MCL | 0.5973 | 0.6570 | +0.0597 | 2 / 946 |
| Medial Meniscus | 0.7718 | 0.8637 | +0.0918 | 412 / 580 |
| Lateral Meniscus | 0.6472 | 0.7344 | +0.0872 | 149 / 821 |
| Medial OA | 0.8825 | 0.9764 | +0.0939 | 23 / 499 |
| Lateral OA | 0.7198 | 0.7727 | +0.0528 | 9 / 531 |
| PF OA | 0.7697 | 0.8616 | +0.0919 | 19 / 445 |
| Effusion | 0.7683 | 0.8830 | +0.1147 | 9 / 671 |
| Synovitis | 0.9504 | 0.9802 | +0.0298 | 121 / 5 |
| Baker's | 0.8316 | 0.8918 | +0.0602 | 3 / 382 |
| Contusion | 0.8182 | 0.9075 | +0.0893 | 61 / 371 |
| Fracture | 0.8494 | 0.9324 | +0.0830 | 2 / 259 |

MCL and fracture each have only two positive held-out components; Baker's has
three and synovitis only five negative components. Their apparent improvements
remain sensitive to very few reports. The leave-one-rare-component calculations
in `comparison.json` measure that sensitivity, not clinical confidence intervals.

All ten immutable epoch checkpoints were each published once to the private
`GodlyDonuts/RSNA-Data` dataset. Every checkpoint, snapshot, license and manifest
was fully downloaded and size/SHA verified: **40 objects**, including
13,381,647,160 checkpoint bytes. `all-epochs-publication.json` and the original
per-epoch intent/commit/download receipts preserve exact revisions and hashes.
Epoch 10 is revision `37065d5fa12296d84e6a7e88354eae9cbda0d3e7`, checkpoint SHA
`3ed9a08f144fb0b3e30dee7bfc65df13f5c4647b2a0532a8d04d3241da0d4db0`.
No A32 competition submission was made; the prior model's best actual public
Kaggle score remains 0.638.

One-CPU final verification **865225 completed 0:0**: all 120 residual modules
learned, all model tensors are finite, all **415 original official base tensors
remain unchanged**, and checkpoint-only strict reconstruction passed. The proof
is `production-checkpoint-verified.json`. It does not claim an additional real-MRI
prediction replay. The first attempt, 865224, failed before reading the checkpoint
because the standalone import path omitted the pinned checkout; the distinct
source-path correction used the exact same verifier. Neither attempt repeats
training or a checkpoint publication.

This result justifies further solo-model development. Next comparisons are the
prepared matched frozen control, replication on independent development folds,
and a source-derived native-detail/slice-context pilot with unchanged accepted
supervision. Fresh storage acceptance is required before another full run:
approximately 22.6 GiB remain under the Lustre soft byte quota, and no additional
GPU job is currently launched. No first-place or private-leaderboard prediction
is inferred from this development-fold improvement.

The complete result/receipt milestone is privately mirrored at HF revision
`de7e6c799bd661dde42e93e707de563b1ae5e0b1`; all **94 objects** were fully re-downloaded and size/SHA
verified. The preserved source milestone is Git `eab7b1f`.
