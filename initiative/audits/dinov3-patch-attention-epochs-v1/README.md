# Patch-attention fold-0 epoch preservation

Production Slurm job **845218** remained running on one H100 and two CPUs at
the September 24, 16:38 UTC check. It had completed all ten training epochs,
with 2,810 finite optimizer updates over the full 2,247-study fold-0 training
set. The fixed-final held-out evaluation had not finished and stderr was empty.
This is training progress, not a development or Kaggle score.

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
existing local dispatch intent or Newton journal. A fixed-final fold-0
comparison to the global-CLS baseline is pending the terminal evaluation.
No patch-attention checkpoint has been submitted to Kaggle.
