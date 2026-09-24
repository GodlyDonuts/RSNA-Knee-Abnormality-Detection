# Fold-2 single-H100 epoch preservation

The full-data fold-2 one-H100/two-CPU job 844305 is running the frozen
ten-epoch DINOv3-L recipe. Its CPU watcher is job 844306. Epoch 1 completed
286 real optimizer updates on 2,281 training studies. The watcher preserved
its immutable `verified_unscheduled_epoch_archive` checkpoint with SHA-256
`5063ba124090041b2cce67fd85b27253bdf03f9dfdb7629753afc3fd4048aeaf`.
This is a training milestone, not a fold-2 validation result or Kaggle score.

The private Hugging Face publisher now checks fold 2's exact 286-update
boundary, 2,281-study count, one-process world size, accepted training Git
`b74060b98f01bff1b924070ccba5ce44dd002f40`, code hashes and official
DINOv3 license. The staged publisher SHA-256 is
`d970a0bbfde03f397fdbd99d18f88600720780fbaf36ea7e1a8c2a71ddfd058c`.
The first epoch was published once under
`experiments/dinov3-v1/training-epochs/fold2-seed20260922/epoch-001` in the
private `GodlyDonuts/RSNA-Data` dataset. All four published files were
downloaded from exact revision `4a9fd0d1edcd2d61bda2d2655001ff288d79e841`
and hash-verified. See [the receipt](epoch-001-verified.json). Later completed
epochs must use the same per-epoch verification before being called preserved.
