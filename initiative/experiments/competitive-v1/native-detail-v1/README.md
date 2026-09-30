# Original-pixel native-detail DINOv3 comparison

The scientific runtime remains `c9fcc7eefb09103942be89bf4fa87410eb9ec5ab`:
rank-32 adapters at all 120 projections, frozen original official DINOv3-L
weights, the same accepted labels/masks/fold, effective batch eight, selected
six-series/16-slice coverage and no neighboring context. The prepared native
production config changes only input side from 224 to 384. Its source cache
must be reconstructed at 384 from original DICOMs, rather than enlarging the
accepted 256 cache. This is separate from rank 64 and whole-series coverage.

CPU job **865808 COMPLETED 0:0** in **14:27**, one CPU/no GPU on evc31, built an engineering cache
for the exact first 16 training and first 16 outer-fold diagnostic studies:
**185 accepted series / 6,105 original slices / 4,543,651,168 original bytes**.
Selection read only development membership, folds and observed masks. It did
not inspect target values, reports, gold or reserved confirmation labels.
The private source revision is `0b5e410288232f570c69ba4f44723356e6ae6ab7`.

Each original object is bound to its pinned private source SHA and size. Every
series is rebuilt at 256 and 384 with the unchanged preprocessing code. Source
UIDs, pixel hashes, slice order, patient geometry, intensity transform and
warnings must match the accepted metadata. The rebuilt 256 counterpart must
remain within the predeclared one-uint16-level engineering comparison band.
That band does not establish clinical insignificance or a model accuracy effect.
The accepted cache remains unchanged. Future native/control training can use
the paired rebuilt caches to hold the reconstruction environment constant.

Both caches passed the actual unmodified training loader on all 32 studies,
with synthetic zero masks for this engineering check. Deterministic archives,
complete member SHA/size maps and exact source/selection/code provenance are
were privately published once and all five objects fully downloaded/hash verified at
**3f23826ff226992204380d34a2956f3805f8b960**. The 256/384 archives are
637,173,760 / 1,383,434,240 bytes, with 370 exact regular members each. Of 185
256 reconstructions, 96 match exactly and the remaining 89 differ by at most one
uint16 level; the cause remains unproven. Publication
requires an absent dedicated path and records its intent before mutation;
an uncertain attempt must be reconciled read-only. Neither these caches nor
the diagnostic configs are competition models.

Large temporary arrays and archives use an owned, private job-local tree and
are cleaned on exit. Small durable receipts have a 128-MiB bound, funded in a
complete fresh Lustre projection leaving 20.501 GiB and 5,977 files at staging.
Allocated CPU probe 865795 measured 767,775,236,096 free bytes on evc31 XFS;
that is a current capacity observation, not durable storage permission or a
capacity promise for another node. Recheck every allocation. The full eligible
cohort has 18,906 series / 643,501 slices: uncompressed uint16 storage alone is
189,776,166,912 bytes at 384, plus metadata, archive overhead and transients.
It cannot be placed in the current Lustre reserve. Plan privately pinned shards
and reproducible owned node-local caches before a complete rebuild.

The native GPU recovery/throughput pilot and full-cohort cache are **not
launched**. Actual allocated H100 gradients, unchanged base tensors, strict MRI
reload, exact full/delta/interrupted recovery and realistic memory/throughput
must pass before production. Compare the same fixed-final 1,156-study outer
fold and every target's observed support; a tiny pilot metric cannot promote
the model. Independent folds/seeds remain necessary for broader claims.
See `comparison-contract.json` and the four prepared, validated configs.
