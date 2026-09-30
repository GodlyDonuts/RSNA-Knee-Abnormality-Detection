# Full-cohort native cache shard plan, not a completed cache

The planner reads only eligible development membership, masks, fixed folds,
accepted series preferences/index and corrected original-source inventory. It
reads no target values, reports, gold or reserved confirmation labels.

The plan covers all 3,403 eligible studies (2,247 training / 1,156 outer),18,906 accepted
series,643,501 slices and449644551848 pinned original DICOM bytes. Seventy-five
study-complete shards have paired 256/384 archive upper bounds below 4 GiB each.
No study is divided, no series/slice is omitted, and all 75 JSON byte hashes and
full membership were independently checked. Largest paired bound4293585158bytes;
largest owned arrays+archives scratch bound9660912140bytes, excluding environment
and the mandatory 20 GiB filesystem reserve. Compression gains are not credited.

The corrected effective inventory is SHA 68d202eb951babc9365eddd2f7a1219201cd7b0905f04afdace785ad0e079537,
compressed SHA 33065e6a60b878e5eda90ff2ffe865b5778cdcfa668adab23f2fd65bccdde240,
at official-source revision 0b5e410288232f570c69ba4f44723356e6ae6ab7. It includes
the two restored source encodings; the historical inventory must not replace it.
Original scientific preprocessing remains C9. Accepted cache/masks/folds stay fixed.

Ten local planner tests passed. This is metadata preparation only: no full cache
arrays were reconstructed, no full-cache worker or GPU production was launched,
and no native accuracy result exists. The full builder and private per-shard
publication/owned runtime restoration remain to be implemented. First complete
the already staged native384 H100 recovery/memory/throughput acceptance, then
use fresh cluster byte/file/scratch projections before full-cache work. Keep
large caches private on owned cluster-local storage and pinned HF; do not stage
75,000 cache files into the tight Lustre inode quota. All large outputs require
full download/hash validation and uncertain mutations require read-only review.

Rebuild metadata with scripts/newton/native_cache_shards.py via runpy from the
repository root, using a distinct output directory and --paired-shard-gib 4.
Selection SHA 8bbf6ed6f2d9ec8e3ab9f39b0ac6f535865b6cb002915af0283c3cf458538ce2.
