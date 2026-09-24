# Exact epoch-8 resilient-series code submission

The owner requested one current DINOv3 submission to establish whether its
Kaggle hidden rerun can execute. The frozen fold-0 epoch-8 checkpoint was used
because its new, private MRI-only deployment package was ready for a guarded
first test. This is **not** evidence that epoch 8 is better than epoch 10. Fold-0
epoch 10 has the fixed-final development metric; its earlier Kaggle notebook
version failed hidden execution and was not reused.

The accepted training checkpoint is SHA-256
`e156bcf1d4e0d7509d9a48ac0630af80e7119ee232e23082371c575ffc207373`.
Training source Git is `b74060b98f01bff1b924070ccba5ce44dd002f40`;
deployment source Git is `df0e283aaaea4a10d77edf6633d598ac4fcb88db`.
The new wrapper uses the official `test.csv` study roster, caps the image cache
at six selected MRI series per study, and retries a bad series using another
series of the same plane. Accepted labels, masks, folds, checkpoint weights and
trained model code were unchanged. A previous private `fix3` Dataset was
created and verified, but its package omitted the cache flags, so no notebook
was pushed or submitted from it.

The distinct `fix4` private Dataset
`godlydonuts/rsna-knee-e008-fix4-e156bcf1d4e0` version 1, ID 12161100, was
created once. All 43 payload files, including the 1.215 GB checkpoint, were
downloaded and hash-verified; see [dataset-verified.json](dataset-verified.json)
and [operation.json](operation.json). The exact private, offline T4 notebook
`godlydonuts/rsna-knee-epoch8-fix4-e156bcf1d4e0` version 1 was pushed once.
Its exact source/settings check and public three-study MRI run completed. The
13-column public CSV had three rows, finite probabilities in [0, 1], and SHA-256
`c5f59cf163e7d845d640eba9c708ce7029d1cca8d44e9baa150df3acb23043fd`.
See [the versioned receipt](public-example-receipt.json) and
[CSV proof](public-example-csv-proof.json). The three-row CSV was **not**
submitted as a competition file.

With Kaggle reporting five live account slots, the canonical single Newton
ledger submitted that saved code notebook **once** at 2026-09-24 01:11 UTC as
competition entry **56507997**. Its exact operation SHA-256 is
`01d51375553a07cef8eb950a8b09b1a9e3e5cc8e9ab3f8c6c2f8565be28341e6`.
The initial read-only [submission status](submission-status.json) has no public
score or hidden-rerun result. A completed three-study notebook is only an
execution check, not a hidden-test score. Hold the remaining four current-day
slots until the authoritative hidden result arrives. The five earlier DINOv3
hidden reruns failed with no scores; this code change addresses a possible,
still unproven failure path.

The sole mutable account ledger remains
`/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3` on Newton. Previous
epoch-8 notebook identities and their uncertainty/history are preserved there;
do not repeat any Dataset, notebook, or submission mutation. Large bundles and
credentials remain private on authorized storage.
