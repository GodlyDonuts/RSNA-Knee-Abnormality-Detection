# Bounded representative cached-pixel review

This separate extension preserves the completed source/header-only handoff.
It contains three ordered cached samples from every series: 15 samples for
030:18, 27 for 036:17 and 15 for 042:11. The seven sheets and three per-case
manifests are under `cache-panels/case-NNN-II/`.

All 19 actual NPZ hashes and arrays, 507 corrected source memberships and
57 pasted display matrices passed verification. Samples are existing normalized
256 × 256 uint16 cache images mapped linearly to uint8. No additional spatial
resize, flip, rotation or crop is applied; literal source tags remain visible.
This is not native-resolution image review, fresh source decoding or clinical
labeling. No network downloads were performed.

`render_cache_sets.py` binds every series and sample to full source rows,
cache/metadata hashes, native fingerprint references, ordered index and source
geometry. `verify_cache_panels.py` separately reloads the corrected inventory
and cache arrays and compares exact pasted matrices. Its complete result is
`cache-panels/display-source-integrity.json`. Each own visual assessment was
personally reasoned and frozen before the parent's visual findings. Later
comparisons must be saved separately rather than overwriting either freeze.

For reproduction, copy the scripts and preserved source/header evidence to a
new sibling staging directory without generated `cache-panels/`, then run
`render_cache_sets.py` and `verify_cache_panels.py` with the workspace Python
environment. Original bound inventory/cache paths must remain available. Both
scripts require new outputs and leave this completed evidence unchanged.
