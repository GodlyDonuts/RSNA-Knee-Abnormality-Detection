# Pinned 3,957-report private checkpoint preparation

Scope is Git `94fed05d353d180e30c968bc0561e05c00ae6880` only. All package payloads are read from the immutable Git tree, and every tar member is checked against its Git blob SHA-1, SHA-256, byte count, original mode and canonical timestamp. The unchanged measured package has 4,464 members, 80 reviewed batches / 3,957 reports, 1,203,620,292 bytes and SHA-256 `849d6a29286f2ee6d1b45e5012bc97407438e71d19990c3dcdefab8079ca2bf4`. Uncommitted correspondence and technical additions are excluded.

The planned operation adds exactly three objects beneath `preparation-checkpoints/94fed05d353d180e30c968bc0561e05c00ae6880/`: `checkpoint.tar.gz`, `manifest.json`, and `README.md`. Full pinned audit evidence is inside the tar. It does not replace the root dataset card or mutable audit paths. No remote write or Git commit has been performed by this preparation task. Training readiness is false.

## Bindings and checks

- `review-plan.json` binds the publisher, builder, immutable measurement, manifest, checkpoint README and exact operation. The caller must supply its reviewed SHA-256, currently `c942d22a9d968f4711d6c794d64b80aa0fcd71c4b42798f60cba178dd7ed04c2`.
- `readonly-preflight.json` records fresh successful CI run `35604527490`, exact Git identity, private HF HEAD `76a4c0dab20fd49ffd0590a4344eac5b67d8be1f`, and both source trees plus five official CSV identities matching corrected source revision `0b5e410288232f570c69ba4f44723356e6ae6ab7`. These read-only results are evidence, not a replacement for fresh execution checks.
- `publisher-vs-prior.diff` shows the implementation changes against the audited compact-head recovery script from the previous checkpoint.
- Offline tests cover exact pinned content, tampered payloads, source membership, disk-floor guards, credential-output suppression, bounded HTTP including `timeout=None`, HTTP-only upload negotiation, privacy, CI, parent races, two-thread publication, reconciliation and partial-success reporting.

## Publication procedure for root review

1. Review the plan, manifest, diff, tests, build receipt and local-preflight result. Preserve the plan digest outside this directory. The tar must exist and have the measured digest before publication.
2. Run the publisher in `--check-local` mode with `--expected-plan-sha` and a new `--output` directory under this staging directory's `runs/`. This is offline and rechecks all package members against pinned Git.
3. Only after root review, run the same publisher with `--execute`, the same plan digest, a fresh run directory and a finite `--max-runtime-seconds` (default 1800, range 60–3600). Use the workspace virtual-environment Python.
4. Require `publication-receipt.json` with `verified_private_preparation_checkpoint`. A returned commit is journaled before verification. Any timeout, failure or changed HEAD requires fresh reconciliation; it does not imply success or authorize a new parent.

Before HF imports, the publisher sets `HF_HUB_DISABLE_XET=1`; the reviewed Hub version and transport file hashes are checked. Every HTTP request has finite connect/read/write/pool caps of 15/120/120/30 seconds, including explicit unbounded overrides; upload uses two threads. Compact HEAD reads request only SHA/privacy. Source trees and all five official CSVs must match before and after. New commits require the exact expected parent; a different HEAD only permits exact verification of the already-published three objects and sources. All LFS identities and ordinary-file bytes are checked.

Disk guards preserve 20 GiB plus 32 MiB operational headroom and 1 MiB for concurrent root metadata. The initial package did not fit; an older tar was independently verified locally and on the private remote as an eviction candidate. This task did not delete it. Later fresh disk measurements showed adequate capacity, allowing the canonical build. Every compressed write rechecks the floor, and incomplete files are preserved as `.partial` rather than passed off as completed packages.
