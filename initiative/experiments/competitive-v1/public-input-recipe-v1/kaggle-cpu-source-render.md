# Kaggle CPU source-bound public input rendering (prepared, NOT run)

Execution-placement optimization only (issue #31). It does not add a producer. The
consolidated producer (`rsna/competitive/public_cache_producer.py`, PR #28) and wrapper
source binding run where the official training DICOMs are already mounted, so the
owning node never acquires 449,644,551,848 bytes / 643,501 objects for a cache of
3403 × `member_bytes()` (≈27.7 GB). Nothing was pushed, dispatched, downloaded or changed
on Kaggle, Hugging Face or Slurm. Protected jobs 899766/908454, live training,
gold/confirmation data and scientific sources are untouched.

## Code

- `public_cache_producer.render_member`: the verify → render → bit-exact `.npz` →
  adapter-replay part of `stage_member`, now factored out with no write.
  `stage_member` behaves the same. `import_member` commits a member rendered elsewhere
  through the same exclusive partial/receipt/link sequence. It runs only after the bytes,
  receipted SHA, role, source digest/revision and arrays all equal the owning intent.
- `rsna/competitive/kaggle_source_render.py`: CPU-only runtime binding, mount resolution,
  verification of the corrected objects' after bytes, the retained parity roster and tensor
  hash, lossless encoding, cohort and shard manifests, owning-node merge, and the shard plan.
- `scripts/kaggle/render_public_source.py`: the notebook entry (`--mode probe|shard`).
  It calls the wrapper's `bind_sources`, `producer_records` and `transport_study` unchanged.

## Source and semantics

- Rosters come from `load_verified_release` plus `load_folds` plus `select_development`
  (fold 0). Both must equal the overlay's fitting/outer hashes (2247/1156).
- `bind_sources` runs over the full 9-part baseline inventory, the exact correction
  plan/receipt and the separately attached **accepted release series CSV**, whose SHA must equal the release
  `series_sha256`. The mounted original `train_series.csv` is independently hash-bound; it is not substituted for reviewed plane metadata. It returns the 18,906 accepted series, and the 25 unaccepted series are
  never opened.
- Both officially corrected objects are hashed at the mount and must equal the plan's
  `after` bytes and SHA (`verify_corrections`).
- Each study's accepted series are copied (bounded, SHA-verified, original `os.scandir`
  order, first-winner ties) into non-output scratch. The producer's `_verify_source` then
  re-checks bytes and DICOM UIDs. The original v15 renderer and the hashed adapter render
  once (130 mm crop, slots, order, laterality, triplets, geometry). The copy is removed
  after each study.
- No labels or clinical text enter pixels or members (`labels_stored: false`). The probe
  reads fitting-only overlay labels solely to reproduce the accepted receipt's tensor hash.

## Probe (root alone dispatches)

The probe covers the retained 8 fitting studies (`parity_roster`, salt
`public-dinov2-input-parity-v1`), rendered from the mount with no download. Per study it
checks the following against the accepted CPU job 906226 receipt:

- original `render_study` + `windows` equals the adapter images and slot mask
  (`original_pixel_triplet_matches` must be 8);
- the 7-key tensor hash over the 8 studies must equal
  `e602f1a976fe341a48794d5cec8919522ee6bdad473770661fb9c1d69db3c6d8`. Ranked and sorted
  roster orders are both reported, because the private packet's roster order is not in Git.

It writes one probe shard (never mergeable), a proposed complete-cohort manifest, and
`summary.json` with sanitized aggregates (counts, hashes, versions, measured
seconds/bytes/RSS/disk, proposed shards).

Root prerequisites, each a root-owned action with identity checks:

1. A private Kaggle Dataset holding this exact revision's code files listed in
   `CODE_FILES`, plus `rsna/`, `tools/` and the needed `initiative/` inputs (config,
   overlay, inventory parquet, correction plan/receipt). It also needs the release
   studies/artifact root and folds, the verified Pilkwang v15 source bytes and the member
   manifest. Existing private wheels/model/CPU fixture assets may be reused only after
   root identity (SHA) checks; no model weights are needed.
2. A private notebook with `enable_gpu: false`, `enable_internet: false`,
   `competition_sources: ["rsna-knee-abnormality-detection"]` and the dataset above. No
   competition submission.
3. A budget JSON (`BUDGET_KEYS`) with `code_sha256` equal to `code_sha256()` of the
   attached files and `runtime_sha256: null` for the probe. Its limits must be root-funded
   from the actual session limits shown on the Kaggle settings page, not taken from this
   document.

```
python scripts/kaggle/render_public_source.py --mode probe \
  --input-root /kaggle/input --private-out /kaggle/working/private/probe \
  --scratch /tmp/rsna-source-transport --budget <budget.json> --budget-sha256 <sha> \
  --source <pilkwang-v15.ipynb> --member-manifest <manifest.json> \
  --studies <studies.parquet> --artifact-root <release-root> --folds <folds.parquet> \
  --accepted-series <accepted-release-series.csv>
```

Probe acceptance requires all of: exit 0, `accepted_parity: true`, 8/8 triplet matches,
`corrected_objects_verified: 2`, the expected source-table totals, `labels_stored: false`,
`source_downloaded: false`. Root then records `runtime_sha256` and the cohort manifest
SHA. If parity fails, no shard runs.

## Shards, storage and merge

- The probe proposes `plan_shards` from its measured slowest study and largest encoded
  member under the budget's wall/output limits, with a 50% margin. The shard size is
  written into the cohort manifest. Each shard requires a budget with the probe's
  `runtime_sha256` and the recorded cohort-manifest SHA. It refuses if the mounted sources,
  code or runtime differ.
- Storage: members are the exact producer `.npz` bytes, `gzip -9` with `mtime=0`
  (deterministic). Decoding restores the identical bytes and SHA, so the tensor, geometry
  and index contract is unchanged. `np.savez_compressed` size is also measured and checked
  lossless. No compression ratio is claimed; only the probe's actual sizes decide whether
  the cohort fits the 20 GB working disk or needs more shards.
- Each shard is immutable: `shard-manifest.json` (cohort SHA, index, sorted UID roster hash
  equal to the assignment, member exact/encoded SHA, receipt/source-order SHA,
  adapter-output digest, runtime) plus per-study receipts and source-order receipts.
- Owning-node merge (`merge_shards`) runs into a cache created by the wrapper's intent for
  the same rosters, sources and revision. It verifies each manifest against its recorded
  SHA, refuses foreign entries, overlaps and probe shards, decodes within bounds, and
  optionally replays the adapter under the owning runtime (digest must equal Kaggle's). It
  then calls `import_member`; already complete members must be byte-identical. After that
  come `reconcile` and the unchanged `finalize`.

## Licensing, privacy, provenance

The competition data and its derivatives stay in private notebook output, the private
owning node and the private cache only. Never put them on GitHub, Hugging Face, public
datasets or public notebooks. Use is limited to this competition's rules. Indices, UIDs,
receipts and source-order files are private. Reports carry only aggregates and hashes.
Transferring private outputs to the owning node is a separate root action. It needs an
admitted storage budget and must also honor existing live-job storage reservations.

## Not claimed

Synthetic tests (`tests/test_kaggle_source_render.py`) exercise engineering only: merge
equals direct staging, refusals, encoding and budget. They do not show Kaggle access,
source parity, runtime fit or accuracy. Each of those needs the actual root-run probe.


## Actual October 13 transport-order recovery

Private CPU order replay138544280/v1 verified1528 original DICOM objects and
reproduced all eight original tensors bit-for-bit (e602f1a9...), including integer
pixels and exact geometry. The cause was directory enumeration: all eight studies
had different directory order, and one of34occupied slots selected a different
series when equal-length candidates tied. Pinning the original Python/Torch runtime
alone reproduced the mismatch; it did not fix it.

Transport now records original and copied series orders. The context-local
`reference.preserve_series_order` replays the original metadata-frame order after
copying, validating complete directory membership and restoring context on exit.
Original source nodes, slot predicates, tie rule, adapter bytes and pixel math are
unchanged. The cohort producer and Kaggle shard entry use that same recorded order.
The original CPU probe was engineering evidence only: no model, optimizer, gold
selection or accuracy gain. Capsule packaging is delegated under issue33; the full
cohort cache and paired training are still not complete.


## Direct-source transport acceptance

The probe now compares every tensor component to the unchanged original renderer
on the SAME mounted original study with the SAME recorded series order, before
transport. This must match per study, not just on average. The acquired historical
e602 hash remains separately reported and cannot be used as a universal source-
order hash: filesystem enumeration selects different equal-length series.
Original source predicates, tie rule and pixel operators remain unchanged.
Full-cohort emission requires actual direct-source parity plus exact object-table,
roster, code/runtime and output-size bindings. Historical-reference disagreement
alone is neither an accuracy result nor permission to weaken source-byte checks.
