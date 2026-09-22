# Guarded epoch publication integration

The three coordinator helpers were copied byte-identically from the reviewed
staging files into `scripts/kaggle/`. Tracked tests adapt only import locations
and a synthetic projection fixture for CI. Training/runtime code and accepted
data are unchanged. Root owns repository-wide checks, archive refresh, commit
and external code publication.

The independent reviewer accepted the corrected source after 32 synthetic tests.
The initial extra-bytecode issue and its source snapshots are preserved alongside
the correction and final review. The integrated Kaggle checks additionally cover
the existing queue, wheelhouse and progress helpers; their actual result is in
`integration-checks.json`.

The actual private structure probe created one Dataset and one offline CPU
notebook. It verified all three logical files, first by streaming their immutable
remote bytes, then inside the notebook mount. The notebook's actual returned slug
was derived from its title; source and settings were reconciled read-only without
another creation. No model checkpoint, medical data, GPU or competition submission
was involved. These receipts do not establish full model runtime or a score.

`integration-mapping.json` binds every original source to its archived byte copy,
plus current tracked helper/test hashes. The archived operational runbook is
`staged-source/README.md`; it describes the prior staged state honestly. The active
tracked runbook is `docs/kaggle-submissions.md`. The corrected runtime baseline is
Git `e7b08625854de692b9266ccfc25c90fc22169f58`, exact-head CI 35766826230 succeeded,
and its HF code overlay was fully verified at
`6f30d9def9b411c3adf2517ec683a22b2adf1bb4`. Both publication receipts are preserved
under `baseline-publication/`.

There is no eligible production snapshot or real competition submission in this
audit. Before deployment, generate a new minimal projection from the packaging
commit containing the helpers and the exact runtime commit of the checkpoint.
A killed/uncertain operation requires read-only reconciliation; never retry a
mutation merely because the remote result is temporarily absent.

The later [bootstrap-v2 supplement](../kaggle-epoch-publication-bootstrap-v2/)
supersedes this audit's current coordinator/test source pins. Root found a clean
CLI import-cache issue before commit; the correction adds four actual fresh-child
regressions and independent validation. The original source snapshots, mappings,
review records and earlier root 937-test result remain historical, unchanged.
