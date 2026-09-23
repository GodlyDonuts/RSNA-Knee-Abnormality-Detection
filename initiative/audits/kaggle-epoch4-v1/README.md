# Scheduled epoch-4 private Kaggle preparation

The immutable fold-0 scheduled epoch-4 checkpoint has SHA-256
`90a72a4f203d233a51ac52ce346d038af125b4cafdfacddafac1ca2d16eb6de6`.
It binds unchanged accepted masks and folds to training Git
`b74060b98f01bff1b924070ccba5ce44dd002f40` and private inference
package Git `2bc26959c4b7d3c82a7cba41f8b8f662970a91ac`.

CPU-only Newton job **841917** built and validated the package. Its Dataset
transport has SHA-256
`8870e99fb55c9e09bf8c9262519da6dd2be2498b7d6489c2bda55240523256b1`.
The private Dataset `godlydonuts/rsna-knee-e004-90a72a4f203d`, ID **12146194**,
was created once as version 1. The first read-only reconciliation hit a Kaggle
API error. A separate direct read confirmed the private Dataset existed; the
second read-only reconciliation downloaded and rehashed every payload file,
including the 1.215 GB checkpoint. No Dataset creation was retried.

The dedicated private notebook `godlydonuts/rsna-knee-epoch4-90a72a4f203d`
was pushed once as version 1. It uses the accepted offline T4 wheelhouse and
MRI-only inference. Its public-example runtime receipt and any competition
submission remain pending. `source-files.json` maps the exact small Newton
files and canonical ledger record; `local-files.json` maps the scripts and
dispatch results. Large private artifacts remain on Newton and Kaggle.
