# Exact Dataset reuse after notebook identity conflict

Original SaveKernel returned HTTP409; the canonical ledger preserved push_uncertain, no kernel version and no competition submission. GetKernel returned403. Complete authenticated owned-profile listing/search, with a known notebook positive control, found no requested notebook identity. The original identity is permanently preserved and never retried.

A single distinct identity godlydonuts/rsna-bestinternal-offline-e10-2465648917b5 was then pushed once, returning exact version1. It reuses the private Dataset12433771/version1 and identical kernel source/checkpoint/manifest/runtime. Only notebook identity and its metadata changed. The canonical ledger retains the old operation and absence proof. The reason for the original409 is unproven. Actual public completion and full output acceptance are still required before submission.
