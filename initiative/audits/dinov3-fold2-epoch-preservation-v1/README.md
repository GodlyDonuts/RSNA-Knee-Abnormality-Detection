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

At the September 24, 02:22 UTC check, the same production job had completed
epochs 2 and 3 and was making real optimizer updates in epoch 4. Epoch 2 has
572 total updates and scheduled checkpoint SHA-256
`092c3eed133f73d7a167c6eb7b55068c2684f82a3714f337839e81874c533605`;
epoch 3 has 858 updates and watcher-preserved checkpoint SHA-256
`420c4f292d00267db5c86ef228f6a3af6341aa98a7a0d35f2d765be992b794da`.
Both were published once to their distinct private epoch paths. All four files
of each were downloaded and hash-verified at revisions
`1521fc8e5ebab9bdbda4b0f6fde784cabf14a332` and
`92fed04b0983904543df5e2c203c4425c0912893`, respectively. See
[epoch 2](epoch-002-verified.json) and [epoch 3](epoch-003-verified.json).
No fold-2 validation metric or Kaggle score is available yet.

The small Git audit milestone at
`370b5f6e31901d6579f30c3aa5b326a5a9c9c127` was separately mirrored to
the private Hugging Face dataset at revision
`935942f6341305640397f3f136fe77be5a073d68`. All six files were downloaded
and hash-verified; see [the milestone receipt](hf-epochs23-milestone-verified.json).
