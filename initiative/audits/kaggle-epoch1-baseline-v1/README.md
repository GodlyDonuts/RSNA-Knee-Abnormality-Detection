# Completed epoch-1 baseline: private Dataset and MRI-only notebook

The owner's one-off baseline uses the completed fold-0 epoch-1 replay checkpoint,
SHA-256 `01d56458c54f4f2811ecfd5b997e99baec765e850345001960b953304ff3d28e`.
Its 281 optimizer steps and all 255 replayed updates were independently checked
against the primary run in the [epoch preservation audit](../epoch1-baseline-and-epoch-preservation-v1/).
The accepted masks, folds, MRI preprocessing, model and trained `rsna/**/*.py`
remain bound to training Git `b74060b98f01bff1b924070ccba5ce44dd002f40`.
The one-off inference adapter and private packaging code are Git
`2bc26959c4b7d3c82a7cba41f8b8f662970a91ac`.

The exact code projection and private Kaggle Dataset preparation passed on Newton.
The Dataset [`godlydonuts/rsna-knee-e001-01d56458c54f`](https://www.kaggle.com/datasets/godlydonuts/rsna-knee-e001-01d56458c54f)
was created **once** as private version 1, ID **12145591**. A separate read-only
reconciliation downloaded and SHA-256 checked **all 46 payload files**, including
the 1,215,174,821-byte checkpoint; `dataset-verified.json` binds the exact files,
operation and version. The accepted private shared wheelhouse and T4 probe remain
separate inputs. No reports, official gold labels or test text entered the bundle.

The first notebook slug, `godlydonuts/rsna-knee-e001-01d56458c54f`, returned an
uncertain push result. Complete authenticated profile listings subsequently showed
it absent. One guarded same-slug recovery omitted the optional nine-hour request
field but received HTTP 409. Both intents and the 409 are preserved; neither
created a visible notebook or a competition submission. A new notebook operation
uses the **same hard-linked Dataset bytes and identical 9,400-byte source** with
a distinct slug, `godlydonuts/rsna-knee-epoch1-01d56458c54f`. Its one guarded
push returned version **1**. Read-only inspection confirmed the exact source
SHA-256 `928fdbd9884f5f803b6093018d17eb92c5cd65d25e5a83e237a75c273681a02d`,
private T4 execution, internet disabled, both intended private Dataset sources,
and the competition source. The notebook is currently running. The server now
applies its default allowed session timeout; the explicit 32,400-second request
was removed during the guarded recovery. No competition slot or score has been
used or claimed at this milestone.

The one canonical Kaggle ledger is
`/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3`. It records the
superseded slug, both uncertain pushes, the byte-bound migration and the
accepted version-1 notebook. The exact small cluster receipts are mapped in
`source-files.json`; code projection, dispatches and recovery scripts are mapped
in `local-files.json`. Large private checkpoint, wheelhouse and Dataset bytes
remain on Newton and Kaggle. The next gate is a successful MRI-only inference
receipt from the completed notebook, followed by one live-quota-checked code
submission of that exact version.
