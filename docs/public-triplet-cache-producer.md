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

Prepared, NOT RUN. Modes `stage`, `reconcile` (read-only) and `finalize`. Requires evc31,
`CUDA_VISIBLE_DEVICES=''` and no scheduler GPU; a root-reviewed funding receipt bound by
SHA256 (`max_total_bytes`, `min_free_bytes`, `max_study_source_bytes`,
`max_studies_per_run`); the frozen config, v15 source, private member manifest, verified
release/folds matching the accepted fold-0 roster hashes, archived inventory and original
raw split. Each study's inventory files are copied with a byte bound and hash check into a
private transport scratch and removed after staging. Output lives under an owner-only
`--private-out` outside Git and outside the retained eight-study packet, which is never the
cohort source and is never re-acquired. Stdout/`--out` carry counts and digests only. It
never dispatches jobs or downloads data; root handles asset, quota and GPU review.
