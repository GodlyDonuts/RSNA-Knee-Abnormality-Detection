# Public ordered-triplet cohort cache producer

Engineering input preparation for the matched public-input DINOv3 experiment
(`initiative/experiments/competitive-v1/public-input-recipe-v1`). Source/code acceptance is
not clinical, accuracy or GPU evidence. The full matched 10-epoch fit still follows actual
source-cohort admission and real pretrained-model recovery.

## API (`rsna/competitive/public_cache_producer.py`)

- `create_intent` validates the exact `public_input` policy (renderer/source/node-text/
  adapter/pixel-group hashes), disjoint unique 2247 fitting / 1156 development membership,
  one canonical original-inventory digest per member and caps that fund every exact member
  byte plus the free-disk reserve, then creates `intent.json` once (`O_EXCL`).
- `Renderer` loads the reviewed v15 source and the hash-bound `dino_adapter.py` bytes once.
- `stage_member` handles one study at a time: inventory digest, exact original file set
  (no symlink/extra/missing file), per-file size/SHA256, DICOM study/series/SOP UIDs, caps
  before render and write, one original `render_inputs` pass, bit-exact `.npz` round trip,
  stored-arrays-to-adapter equality, then an exclusive staged write, an immutable receipt,
  a hard link into `members/` and removal of the staged file. A completed or receipted member
  is verified and reused, never re-rendered or overwritten.
- `reconcile` is read-only; it reports `pending`, `stale_partial`, `receipt_unlinked`,
  `linked_with_partial`, `complete` and `corrupt` members and unexpected entries.
  `release_stale_partial` discards only an unreceipted partial.
- `finalize` writes the unchanged trainer `index.json` once, atomically, only when every
  member is complete with verified bytes, nothing foreign exists and the total is within
  the funded bound; it re-reads the index through `public_input.load_public_index`.

Members hold only `images`/`slot_mask`/`features`/`positions` with exact dtypes/shapes; no
labels or label masks. The trainer supplies those from its unchanged frames. Native or float
caches are refused. Default trainer behavior is unchanged.

## Owning-node wrapper (`scripts/newton/build_public_triplet_cache.py`)

Prepared, NOT RUN; it never dispatches jobs, downloads, deletes or alters original data.
Modes `stage`, `reconcile` (read-only) and `finalize`; the producer above is unchanged.

- Placement: evc41 CPU-only is the supported placement; evc31 is accepted only as an
  `independently_funded_alternate`. `CUDA_VISIBLE_DEVICES=''`, no scheduler/step GPU
  variables (`SLURM_JOB_GPUS`, `SLURM_STEP_GPUS`, `GPU_DEVICE_ORDINAL`, nonzero GPU counts)
  and `torch.cuda` must report no device.
- Funding (schema 2, bound by `--funding-sha256`): `node`, `placement`, `cpu_only=true`,
  `step_gpus=0`, `wrapper_sha256`, `producer_sha256`, `source_table_sha256`,
  `runtime_sha256` (canonical runtime receipt), `max_total_bytes`, `min_free_bytes`,
  `max_study_source_bytes`, `max_studies_per_run`. Any node/placement/hash mismatch refuses.
  The intent binds the funding hash, so changed funding cannot resume an existing cache.
- Sources: the complete 9-part baseline inventory, the exact correction plan and receipt
  (`initiative/audits/source-pixel-representation-v1`), the release series CSV (SHA must equal
  the verified release) and the 2247/1156 fold-0 rosters go through
  `public_source_table.accepted_source_objects` at corrected revision `0b5e410`. Its 8-column
  objects become producer 6-field records only after split/study/series/SOP/path identity
  checks. The intent records the source-table summary and every input hash.
- Retained packet (`--retained-packet`, read-only): a study is reused only if every accepted
  object equals the packet's historical `52f995` object (path/SHA/bytes), the packet roster
  matches its admitted plan hash and no accepted series of that study is touched by the
  correction plan. Everything else must come from `--dicom-root`, already acquired by root at
  `--source-revision 0b5e410`; every copied file is SHA/byte verified.
- Transport: per study, only accepted series are copied (bounded, `O_NOFOLLOW`, hash checked,
  read-only copies) into `--private-out/transport` before the producer's header/slot/
  laterality probe; unaccepted series are never opened. Symlinks, traversal, extra or missing
  files and source changes during copy refuse. The copy is removed after every study, and
  leftovers of an interrupted run are discarded (symlinked leftovers refuse).
- Source order: original `read_headers` iterates `os.scandir` and `pick_slots` keeps the first
  row among equal `n_slices`. The copy's scandir series order must equal the filtered original
  order or the study refuses. `source-order/<study>.json` (create-once, intent-bound) records
  origin/revision, series order, n_slices and tie groups; a later differing order refuses.
  Parity is only claimed for this owning-node order, not for any reordered source.
- Staging requires a clean reconcile (only pending/complete, nothing foreign, not finalized).
  `--private-out` is owner-only and may not overlap Git, the retained packet or any source
  root. Stdout/`--out` carry counts and digests only. Full acquisition stays root-owned.

Supersedes the producer/wrapper drafts of PRs 19-23 (PR 24's binder is already on main).
