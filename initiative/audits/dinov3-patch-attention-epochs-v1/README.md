# Patch-attention fold-0 epoch preservation

Production Slurm job **845218** completed with exit `0:0` on one H100 and two
CPUs. It finished ten training epochs and 2,810 finite optimizer updates over
the full 2,247-study fold-0 training set. The fixed-final held-out evaluation
produced [final-result.json](final-result.json), downloaded from Newton with an
exact matching SHA-256 of
`1d59324038753f3dd9708e32f32bdc7be0c15d54c2e7fc5bf39a559bcf7ad134`.
Its macro AUC was **0.7451169387** on 1,156 fold-0 holdout studies, versus
**0.7783102649** for the same fold's global-CLS DINOv3-L baseline. MCL AUC
was 0.2162 versus the baseline's 0.5973, a conspicuous target-specific
regression to investigate before repeating this head on other folds. These
are internal development metrics, not Kaggle scores.

Each immutable completed epoch 1–10 was published exactly once to the private
[`GodlyDonuts/RSNA-Data`](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data)
dataset under `experiments/dinov3-v1/training-epochs/patchattention-fold0-203f419-seed20260922/`.
The pinned publisher re-downloaded all four files of each epoch and checked
their full byte counts and SHA-256 hashes. The verified private revisions,
checkpoint hashes, and source binding are in
[verification-receipts.json](verification-receipts.json); the matching Newton
checkpoint sidecars are in [snapshot-sidecars.json](snapshot-sidecars.json).
The most recent verified revision is
`0b2f3d85471267cf86cfc7eb301a2a7d3ecabc4d`.

All ten epoch publications are complete. Do not repeat an epoch with an
existing local dispatch intent or Newton journal. Matched fold-1/fold-2
patch-attention runs are on hold pending diagnosis of the fold-0 regression.
No patch-attention checkpoint has been submitted to Kaggle.
