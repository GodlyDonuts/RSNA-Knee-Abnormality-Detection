# Fold-1 ten-epoch private preservation

Fold-1 training job 843935 and its odd-epoch watcher 843936 completed all ten
epochs. The fixed-final development holdout macro AUC is 0.7128332913 across
twelve accepted report-derived targets on 1,125 outer-fold studies; this is
not a Kaggle score and its holdout differs from fold 0's. Training weights,
source Git `b74060b98f01bff1b924070ccba5ce44dd002f40`, accepted masks and
fold policy are unchanged from the ten-epoch recipe.

Each immutable completed epoch 1–10 was published exactly once to the private
[`GodlyDonuts/RSNA-Data`](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data)
dataset under `experiments/dinov3-v1/training-epochs/fold1-seed20260922/`.
The publisher downloaded all four files of each epoch back from its exact
revision and checked every hash, including the checkpoint and license. All ten
`private_epoch_fully_verified` receipts, revisions, paths and checkpoint hashes
are in [verification-receipts.json](verification-receipts.json). The final
publication revision was `659f28937a9fd52027cde968610e498913486fd3`.
No fold-1 checkpoint has been submitted to Kaggle.
