# Patch-attention fold-0 epoch preservation

Production Slurm job **845218** remained running on one H100 and two CPUs at
the September 24, 14:42 UTC check. It had completed five of ten epochs, with
1,405 finite optimizer updates over the full 2,247-study fold-0 training set;
the next epoch was making finite progress and stderr was empty. This is
training progress, not a development or Kaggle score.

Each immutable completed epoch 1–5 was published exactly once to the private
[`GodlyDonuts/RSNA-Data`](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data)
dataset under `experiments/dinov3-v1/training-epochs/patchattention-fold0-203f419-seed20260922/`.
The pinned publisher re-downloaded all four files of each epoch and checked
their full byte counts and SHA-256 hashes. The verified private revisions,
checkpoint hashes, and source binding are in
[verification-receipts.json](verification-receipts.json); the matching Newton
checkpoint sidecars are in [snapshot-sidecars.json](snapshot-sidecars.json).
The most recent verified revision is
`261a56f2ba825ad30519a94fb8be66308d5b6db2`.

The remaining epochs need the same one-time publication after their snapshots
complete. Do not repeat an epoch with an existing local dispatch intent or
Newton journal; reconcile any uncertain result read-only. A fixed-final
fold-0 comparison to the global-CLS baseline is pending the full training run.
No patch-attention checkpoint has been submitted to Kaggle.
