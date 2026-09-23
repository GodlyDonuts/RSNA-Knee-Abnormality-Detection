# Scheduled epoch-10 private Kaggle Dataset verified

The immutable final fold-0 scheduled checkpoint has SHA-256
`1464f64063d2ac8729452f0f3edc112a21516ce51ad3b8718a176a599dc3f28e`.
It is the same checkpoint bound to the completed primary trainer result and
private Hugging Face epoch-10 revision. Accepted masks, folds and training Git
`b74060b98f01bff1b924070ccba5ce44dd002f40` are unchanged; the exact
inference package is Git `2bc26959c4b7d3c82a7cba41f8b8f662970a91ac`.

CPU-only Newton job **842239** prepared and validated the package. Its Dataset
transport has SHA-256
`0892d8d355cff9b8eed7a3fc1f9d041b7b6a98ade1b03c9aed346afedbfb54da`.
One private Dataset creation request for
`godlydonuts/rsna-knee-e010-1464f64063d2` returned `Ok`. Kaggle initially
returned HTTP 403 and omitted the slug from the owned-private listing. A later
read-only `GetDataset` exposed private version 1, ID **12146648**. All 43
payload files, including the checkpoint, were then fully downloaded and
rehashed against the original publication intent. No Dataset creation retry
occurred.

The first notebook-push dispatch stopped during preflight. The canonical ledger
is still `queued`, before a durable push intent or `SaveKernel` call. Kaggle's
owned notebook listing contains one item with no `ref`, so the exact notebook
slug cannot be proven absent by the current guard. No notebook push or
competition submission has occurred for epoch 10. Reconcile that listing before
a new notebook mutation; the epoch-10 checkpoint is also fully verified on
Newton and private Hugging Face.

`source-files.json` maps the exact small Newton files and publication
intent/response/verification; `local-files.json` maps the scripts and dispatch
results. The two reconciliation files record the bounded Kaggle checks.
Large checkpoint and transport bytes remain on Newton.
