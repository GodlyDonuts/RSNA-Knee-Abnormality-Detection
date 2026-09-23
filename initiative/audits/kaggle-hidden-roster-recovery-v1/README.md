# Corrected hidden-test roster package, September 23

All five first-round Kaggle hidden reruns failed without output or scores; see
[the failure audit](../kaggle-hidden-rerun-failure-v1/). The reason is not proven
because Kaggle exposed only a generic error. The first-round notebooks used the
three-row public `sample_submission.csv` as their inference roster, although
Kaggle replaces `test.csv`, `test_series.csv` and `test_series/` for scoring.

The corrected package uses `test.csv` as the ordered study roster and treats
`sample_submission.csv` as optional schema evidence. The trained `rsna/**/*.py`
tree, completed epoch-8 checkpoint, accepted labels/masks, and fold are unchanged.
The exact 51-file Git projection is pinned by [its receipt](projection-receipt.json)
to package commit `768f9ca4b536427ccdffc3a60b4a9fa60e602f6d` and training
commit `b74060b98f01bff1b924070ccba5ce44dd002f40`. Its manifest SHA-256 is
`943f286a36bc0a5b5d4432fbb1149ff536e7e47dd1798e205a3e87a999598945`.
All projected files and the archive were independently hashed after staging at
`/home/sa305415/rsna-knee/kaggle/code-projections/768f9ca-b740-hiddenfix-v1`.

One CPU-only Slurm job **842807** on evc42 completed `0:0` in 46 seconds and
prepared the new private-cluster bundle at
`/home/sa305415/rsna-knee/kaggle/bundles/epoch008-e156bcf1d4e0-768f9ca-hiddenfix-v1`.
The pinned [operation](operation.json), [manifest](bundle-manifest.json),
[private notebook source](run.py.txt), [metadata](kernel-metadata.json), and
[job output](job-stdout.txt) are preserved here. The bundle retains the
1,215,174,821-byte epoch-8 checkpoint SHA-256
`e156bcf1d4e0d7509d9a48ac0630af80e7119ee232e23082371c575ffc207373`.
The local stage verification checked the new chunk wrapper SHA-256
`f6d6224233e5ecbb3b990ccc592f037409f203cc9de4c2760ec0dddeb25503d0`
and the notebook's `--test-studies` input. The new Dataset and notebook slugs
are `godlydonuts/rsna-knee-e008-fix2-e156bcf1d4e0` and
`godlydonuts/rsna-knee-epoch8-fix2-e156bcf1d4e0`.

This is **prepared and locally validated only**. It is not yet a published
Dataset, a completed Kaggle notebook version, a competition submission, or a
hidden-run fix proven by a score. The original epoch-8 version 1 remains held.
Publish the corrected Dataset under a distinct immutable journal, fully
download/hash verify it, then push and validate one distinct private offline
notebook under the single Newton account ledger. Submit only after a fresh live
allowance and inspect that hidden run before using another slot.
