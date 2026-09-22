# Guarded direct-release promotion

This is a prepared, additive, private HF publication operation. Preparation and tests performed no remote writes. Root owns execution. Git A is `f41879a33b38e47f0cbd96025edac7e2176a0a2f`; the required verified checkpoint HF parent is `c0b2c8b041120ae6d27891145f1a19e48a6d498f`, bound to actual root receipt SHA-256 `85eff309806b04c658ec59001607a585be2603eb2e7d7337275a6a49c9ac17c8`.

The exact operation adds 19 objects under `releases/report-supervision-v1/`: 11 unchanged Git release files, six unchanged supporting files, and generated `PUBLICATION.json`/`README.md`. Total upload is 139,004,559 bytes. There are no root-card, source, CSV, policy, Git or existing-version edits. The source payloads are read from pinned Git objects into memory, avoiding an extra 139 MB disk copy. The 20 GiB free-space floor remains enforced.

The prepared operation hash is `e665e15adbee27e969c4a98ecb42a5cc23bc20a45f2625c6f054b586008f4455`. Review `operation.json`, `PUBLICATION.json`, the generated release `README.md`, and `promote_release.py` before execution. Use the same Python environment as the existing checkpoint publisher: its exact reviewed Hub 1.18.0 transport is required.

```sh
python3 work/annotation-staging/final-direct-release-publication-v1/promote_release.py publish \
  --actual-executor /root \
  --operation-sha256 e665e15adbee27e969c4a98ecb42a5cc23bc20a45f2625c6f054b586008f4455
```

Only the exact latest checkpoint parent and an absent version prefix permit a new commit. The helper checks private visibility, fresh exact-commit CI, protected train/test tree and official CSV identities, root-card identity, local payload hashes and upload identities, then uses a compare-and-swap parent and two upload threads. An exclusive local `root-mutation-attempt.json` is written before the commit call. Do not remove or reset that marker after an uncertain result.

For a retry, change `publish` to `reconcile` in the command. This action never writes remotely. A repeated `publish` after the marker exists is also forced to read-only reconciliation. If the expected release is absent or incomplete, the retry fails closed; root must investigate rather than overwrite an existing version or delete the marker. Exact content at a changed HEAD can be reconciled without asserting authorship or ancestry.

The original checkpoint transport validates LFS object SHA and size, and streams ordinary objects; the distinct prior root and peer restorations establish the complete Git tree. This promoter additionally streams all 19 immutable promoted objects, including LFS payloads, and verifies every byte's SHA and size. It then verifies exact prefix membership, continued private HEAD, and unchanged source/card. Only this completes `root-promotion-receipt.json`. No full checkpoint-tar redownload is claimed.

The direct folder is a convenient data download, not a standalone runtime closure. Restore the pinned full Git tree/checkpoint to obtain code, all 1,240 sealed runtime inputs, original reviews, reports, and diagnostic outputs, and preserve the original dependency paths. PUBLICATION explains masking, exclusions, historically exposed gold, weak validation, source-origin limitations and local cache rebuild. Readiness is data/software acceptance, not measured competitive performance.

Validation: 17 offline tests, including altered receipt/CI/source/card, changed dependency pins before imports, exact object membership, no-overwrite, read-only uncertain retries, exclusive attempt marker, memory/disk bounds, LFS corruption/excess/deadline and full-byte verification; Ruff passed. The independent subagent `/root/competition_contract/medical_encoder_sources` reviewed final code SHA `ee0e81934a5d481148ecca4fb31f2790838ea23912f9881cf50ff77a479e7d3c` and accepted the corrections it requested. Its review ran no remote actions and edited no files.

Only source `.py`, `.json`, `.md` files in this stage are artifacts; do not archive Python bytecode caches. This helper intentionally uses the original staging layout and pinned predecessor helpers. Preserve it as historical execution evidence with those exact dependencies; do not move and execute it from a different path without a separately reviewed successor.
