# Public ordered-triplet streaming cohort cache producer v1 (prepared, NOT RUN)

Issue #16 follow-up (comment 6146902333). Replaces the dict-of-all-studies route for
cohort production; `write_public_triplet_cache` is unchanged for existing callers/tests.
Synthetic engineering only: no cache was produced, no job dispatched, no GPU, no
source acquisition, no accuracy or clinical claim. 899766/905334/frozen outputs and the
retained eight-study packet are untouched.

## Producer (`rsna/competitive/public_input_cache.py`)

- Before any intent: exact disjoint unique accepted rosters with expected counts
  (2247/1156 in the wrapper), and every original object (`train`, canonical
  `train_series/{study}/{series}/{sop}.dcm`, size, SHA256, storage, unique path/SOP) for
  the complete cohort, inside per-study/total source caps.
- Intent (`intent.json`, exclusive) binds renderer/source/node-text/pixel group/adapter
  /plane-map hashes, source and raw revisions, roster/cohort hashes, inventory binding
  hash, caps and wrapper provenance. Labels/masks are never stored.
- Streams in sorted UID order: per-study storage check, bounded fetch of that study only
  into private scratch, per-object size/SHA256 and DICOM header identity, render through
  the exact hashed adapter `render_inputs` (its `study_inputs` validation runs; the
  uint8 source row it receives is stored verbatim), contract check, member cap,
  exclusive `members/{uid}.npz` (fsync), then an fsynced journal line. Scratch for the
  study is removed before the next one; at most one study is resident.
- `index.json` is staged exclusively and hard-linked only after every journaled member
  is re-verified and the cohort hash equals the intent; it is the existing
  `load_public_index`/`verify_public_cohort` format.
- Failures raise `ProducerFailure(stage, error_type, completed)` with no IDs, paths or
  messages and leave an index-less partial cache.
- `reconcile()` is read-only. Resume requires an explicit flag and an identical intent;
  completed members are verified and never rewritten or rendered again. An unjournaled
  member file, a torn journal record, a staged index, changed bytes or a differing
  intent refuses resumption for manual review.

## Transports

- `LocalTransport` (used by the wrapper): bounded copy from the retained owning-node
  inventory root; no symlinks, no path escape, stops past the inventory size, removes
  partial copies, never overwrites.
- `ResolveTransport`: bounded pinned-revision streaming client with fixed failure
  categories (no URL/credential/study in errors), no retry. Not used by the wrapper.

## CPU wrapper (`scripts/newton/run_public_triplet_cache.py`)

Refuses unless a SHA256-pinned binding matches: owning node, no visible GPU or nested
scheduler submission, `OMP_NUM_THREADS=1`/one CPU, runtime versions, clean Git HEAD and
module bytes, every input file, private absolute output/scratch outside the checkout,
fresh scratch, explicit resume decision and free-space envelope. Rosters come from
`load_verified_release`/`load_folds`/`select_development(..., 0)` and must equal the bound
roster hashes. stdout/`--out` contain only sanitized counts, hashes and provenance.

Required before any actual run: fresh original asset/job/funding/storage review, the
bound binding file, and the separately repaired owning-node adapter check (not claimed
here).
