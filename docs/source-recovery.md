# Exact authoritative source recovery

`rsna.source_recovery` builds a new inventory and cache from an accepted,
independently assessed two-file source replacement. The original pinned source
revision, inventory, cache, and decoder remain unchanged. Successful recovery
still requires a separate complete image audit and every label-release gate.

The observed mirror defect affected one slice in each of two series. Both mirror
files retained 16-bit pixel headers while storing one byte per sample. Fresh
authenticated Kaggle originals contain the complete valid 16-bit representation;
their decoded values and nonpixel metadata were independently compared with the
mirror. The recovery uses the exact official bytes. It never guesses layout,
changes DICOM tags, pads a raster, or adds a decoder exception.

## Inputs and validation

The command accepts the immutable `source_correction_plan.json` and a separate
`verified_private_hf_source_correction` receipt. Both assessor identities must be
distinct. The receipt must bind the plan, corrected inventory, exact two remote
DICOM hashes/sizes, repository privacy, verified remote objects, and a corrected
commit distinct from the baseline and publication parent.

The full baseline and corrected inventories must agree in every field except
the two explicitly listed source SHA256/byte-size pairs. Complete before/after
rows, all source identities, canonical inventory fingerprints, counts, byte
totals, and all plan evidence hashes are checked. Original replacement files
must match their SHA256, size, CRC, UID triplets, decoded shape, and dtype. The
local effective source tree must contain exactly every slice in both affected
series; every file's hash, size, and UID triplet is checked.

For each unchanged series, the existing cache metadata must cover exactly the
unchanged inventory paths, SOPs and source hashes and match the baseline revision
and current cache version. Each NPZ is streamed through SHA256 and size checks
before reuse. A new hardlink preserves those NPZ bytes without another full
cache allocation. JSON metadata is copied to a separate inode and only its
`source_revision` value changes. A separate lineage row records the original and
new metadata hashes, unchanged cache hash/size, and both revisions. The two
changed-source series are always freshly decoded in full, including their 72
unchanged slices, with the existing strict `cache_series` implementation.

This is an offline verifier of the preserved publication receipt; the tool
itself does not contact Hugging Face or independently repeat remote publication.
It verifies cached bytes and their original source membership, not a second
native decode of the other 819,561 slices. The mandatory full audit checks every
effective cache array and reconstructs the full duplicate/leakage graph.

## Run the preserved correction

From the recovered repository, with `work/` as its parent:

```bash
python -m rsna.source_recovery \
  --baseline-inventory initiative/source/inventory \
  --baseline-cache ../cache256-v3 \
  --patch ../source-byte-failures-v1/source-correction-publication/source_correction_plan.json \
  --receipt ../source-byte-failures-v1/hf-source-correction-receipt.json \
  --artifact-root .. \
  --official-source-root ../source-byte-failures-v1/effective-native \
  --output ../cache256-v3-official-recovery-v1 \
  --min-free-gib 20
```

`--artifact-root` resolves preserved evidence and official originals referenced
by the plan. The corrected inventory path resolves beside the plan itself. The
effective source tree retains the complete `train_series/study/series/SOP.dcm`
layout. Hardlink reuse requires the new cache directory on the same filesystem
as the baseline cache. The default free-space reserve is 20 GiB and the command
uses two Torch threads and sequential fresh-series decoding.

The new output contains:

- `inventory_effective.parquet`: exact bytes of the accepted corrected inventory.
- `cache/`: every unchanged NPZ plus copied metadata, and both fresh series.
- `cache_recovery_lineage.jsonl`: one explicit reuse or fresh-decode row per series.
- `source_recovery_status.json`: bound inputs, coverage, progress, failure or
  completion state, and mandatory follow-up. `training_ready` and
  `full_audit_complete` remain false.

For the accepted correction, expected completion is 24,384 reused series,
2 freshly decoded series, 24,386 total series and 819,635 slices. The corrected
source revision is `0b5e410288232f570c69ba4f44723356e6ae6ab7`; the original
`52f995f4568e36bfa5fbbaf69a89af41226b4f5d` remains preserved.

The command only creates a new directory. Input/preflight failures create no
output. Failures after creation preserve partial files and a failure status;
an abrupt process interruption can leave status `in_progress`, which is also
incomplete. An existing output is never overwritten or resumed. Preserve and
inspect a failed directory, then use a different new output for a retry.

Treat both cache trees as immutable. NPZ hardlinks share their data inode:
in-place editing either name would change the other tree. Never edit or rewrite
these NPZ files in place; any later transform must use a separate output.

After successful recovery, run a new complete audit:

```bash
python -m rsna.audit \
  --inventory ../cache256-v3-official-recovery-v1/inventory_effective.parquet \
  --cache ../cache256-v3-official-recovery-v1/cache \
  --revision 0b5e410288232f570c69ba4f44723356e6ae6ab7 \
  --output ../dicom-audit-v3-official-recovery-v1
```

Use this effective inventory, cache, revision, and resulting audit consistently
for later exact-series selection and label-release validation. Older surveys
and audits retain their original bindings. Their scope cannot be changed by
simply replacing a revision string.

## Verification

`tests/test_source_recovery.py` exercises a real three-series synthetic DICOM
corpus with two malformed mirror slices and two authoritative replacements.
It verifies full fresh decoding, exact cache hardlinks, independent metadata
inodes, unchanged original bytes, complete counts and a separate successful
audit. Adversarial cases include incorrect revisions/UIDs, altered source or
evidence hashes, extra inventory changes, self-approval, duplicate changes,
missing source slices, stale cache metadata, same-size NPZ corruption, reuse of
a stale affected-series cache, interrupted decoding, existing outputs, and a
failed disk-space guard.
