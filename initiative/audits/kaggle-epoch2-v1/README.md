# Scheduled epoch-2 private Kaggle preparation

The primary fold-0 epoch-2 checkpoint is the immutable scheduled snapshot with
SHA-256 `f56c0d64519fbf2fde5433ef4ff3dcc59d8f01394d80515ebf5cebf97726fa02`.
The accepted report-supervision masks, folds and training Git
`b74060b98f01bff1b924070ccba5ce44dd002f40` are unchanged. The exact
inference package is Git `2bc26959c4b7d3c82a7cba41f8b8f662970a91ac`.

CPU-only Newton job **841763** built and validated the private package from the
epoch-2 snapshot. Its Dataset transport has SHA-256
`2e67eb063a4e32f12b365f0bc22fbad1bd82f3494dff41d239e486c9d523307c`.
The private Dataset `godlydonuts/rsna-knee-e002-f56c0d64519f`, ID **12146012**,
was created once as version 1. An independent read-only reconciliation downloaded
all payload files, including the 1.215 GB checkpoint, and verified their bytes
against the bound publication intent. The distinct private notebook slug
`godlydonuts/rsna-knee-epoch2-f56c0d64519f` was pushed once as version 1. It
completed MRI-only inference for the three public example studies using the
accepted shared offline wheelhouse and a T4; its version-bound receipt passed.
The canonical ledger then submitted that exact notebook version once as Kaggle
entry **56483451**, after a fresh live allowance of four. The Kaggle score and
hidden-test runtime remain pending. The three-row public output is not a
leaderboard score.

`source-files.json` binds the exact small Newton files and the one canonical
ledger records before and after submission. `local-files.json` binds the dispatch scripts,
intents and results. The large checkpoint, wheels and Dataset remain on private
Newton and Kaggle storage.
