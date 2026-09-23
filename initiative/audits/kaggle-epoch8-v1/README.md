# Scheduled epoch-8 private Dataset and uncertain notebook push

The immutable fold-0 scheduled epoch-8 checkpoint has SHA-256
`e156bcf1d4e0d7509d9a48ac0630af80e7119ee232e23082371c575ffc207373`.
It binds unchanged accepted masks and folds to training Git
`b74060b98f01bff1b924070ccba5ce44dd002f40` and private inference
package Git `2bc26959c4b7d3c82a7cba41f8b8f662970a91ac`.

CPU-only Newton job **842231** built and validated the package. Its Dataset
transport has SHA-256
`031122adb494627344119537f1658a3b7a9ed38553f62bea607dee8f6249d0a6`.
The private Dataset `godlydonuts/rsna-knee-e008-e156bcf1d4e0`, ID **12146499**,
was created once as version 1. A separate read-only reconciliation downloaded
and rehashed every payload file, including the 1.215 GB checkpoint.

One guarded push of the dedicated private notebook
`godlydonuts/rsna-knee-epoch8-e156bcf1d4e0` returned a Kaggle server error.
The canonical ledger records `push_uncertain`; there has been no second push
or competition submission. Subsequent read-only checks found no entry in the
complete authenticated owned-notebook listing and `GetKernelSessionStatus`
returned HTTP 404, while `GetKernel` returned HTTP 500 for this slug. Those
responses do not establish a valid completed notebook. Preserve the original
intent and reconcile the slug before any new notebook mutation. The epoch-8
Dataset and checkpoint bytes need no reupload.

`source-files.json` maps the exact small Newton files and canonical ledger
record; `local-files.json` maps scripts and dispatch results. The separate
`reconciliation-readonly-20260923.json` records the bounded Kaggle checks.
Large private artifacts remain on Newton and Kaggle.
