# Bounded original-DICOM shard worker and restoration checks

Source milestone GitHub main `36eedd6` is privately mirrored at HF revision
`a69d588da7bced8224b5340b38696da07a4ce210`. All seven source/evidence objects
were fully re-downloaded and size/SHA verified. The original once-only intent,
commit and complete verification receipts are retained here; never repeat it.

`scripts/newton/build_native_cache_shard.py` implements the next CPU cache
worker. It builds both256 and384 directly from corrected pinned original
DICOM bytes, without an intermediate resized image. It validates every object
size/SHA and UID, complete slice membership, source pixel fingerprints, geometry,
ordering, spacing, intensity and warnings against the accepted256 cache metadata.
The rebuilt256 arrays must match within one uint16 level. Larger differences
stop the shard; the cause of tolerated rounding differences remains unproven.
No original array, report, target, mask or fold is changed.

The actual corrected75-shard metadata/source-membership validation is preserved
here. This covers3403 eligible studies,18906 accepted series and643501 slices.
It is metadata validation, not a fresh source download or full array rebuild.
The worker pins the exact full-plan manifest and verifies its original input
bindings, corrected inventory and per-shard bytes before allocating output.

Local tests use real synthetic DICOM encodings, source geometry ordering and
both cache sides through the unmodified MRI loader. They also restore each
archive with complete byte/member validation and reject source/reference
corruption, missing/duplicate/wrong-study membership, geometry/intensity/pixel
changes, errors outside the one-level band, unsafe paths, links, duplicate tar
members, existing output and underestimated archive bounds. The28 cases passed.
These tests are engineering evidence; no clinical prediction score is claimed.

Production CLI requires a one-CPU/zero-GPU Slurm allocation, at least20GiB
node-local free reserve beyond planned owned scratch, a fresh private output
outside Lustre, the exact C9 checkout and hashes of the actual imported
scientific modules, private pinned HF source metadata and the original accepted
reference archive. Supply the local credential through stdin only. The worker
records actual runtime/worker/scientific/plan bindings. Invoke the external
worker with runpy from the pinned checkout so its imports bind to C9.

Large arrays and paired archives remain in the owned node-local directory.
Per-series proofs use one JSONL per shard instead of separate Lustre files.
Both archive member maps and full hashes are retained; verification includes
all bytes and optional fresh extraction without tar.extract or tar.extractall.
This extraction helper does not yet implement multi-shard/private-HF assembly.

No worker Slurm job, full-cohort cache build or publication was dispatched for
this milestone. Private per-shard publication, orchestration, failure recovery,
and full-cohort training-time restoration remain outstanding. Complete the
already-staged native GPU recovery/memory/throughput acceptance when an actual
GPU role finishes, then run a bounded real-source worker acceptance before
rebuilding449GB of original sources. Never restart an existing intent blindly.
