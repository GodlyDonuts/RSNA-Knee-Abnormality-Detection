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
or competition submission. `GetKernelSessionStatus` returned HTTP 404 and
`GetKernel` returned HTTP 500. A later exact authenticated search returned one
unnamed `[Private Notebook]` placeholder for the old slug. It is not a valid
completed version and must never receive a repeated `SaveKernel` call.

A second *notebook identity*, `godlydonuts/rsna-knee-epoch8-k2-e156bcf1d4e0`,
is prepared from the same version-1 private Dataset and byte-identical MRI-only
notebook source. Its 44 source Dataset files are hard-linked, so no checkpoint
or Dataset payload was copied or uploaded. The derived operation SHA is
`dbb632e7e3afffbbc9324eb484f4cf4185fe0142abb71d68d6042c3998cb5ce1`.
The first derivation preflight counted the separate Dataset metadata sidecar as
a payload receipt member and stopped before linking any bytes. The empty
temporary directory was removed after an exact contents check; the corrected
second derivation passed. The reviewed exact-search guard found a known owned
notebook as a positive control and proved the new slug absent. One CPU-only
migration recorded a durable intent, preserved the uncertain old slug in the
canonical ledger, and pushed the unchanged MRI-only notebook as **private
offline version 1** under the new slug. Kaggle completed its three-study
public-example run and returned a version-bound MRI-only receipt. The canonical
ledger remains `pushed`, without a competition submission. A fresh live account
check showed five submissions today and **zero slots available**. The exact
notebook is ready to submit when a new authoritative allowance appears; no
quota reset time is asserted. The public three-study output is only a package
check, not a leaderboard score.

`source-files.json` maps the exact small Newton files and canonical ledger
record and derived operation; `local-files.json` maps scripts and dispatch results. The separate
`reconciliation-readonly-20260923.json` records the bounded Kaggle checks.
`public-example-receipt.json` and `submission-readiness.json` bind the completed
version and zero-slot account observation. Large private artifacts remain on
Newton and Kaggle.
