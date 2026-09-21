# Representative cache review of mixed side tags

The three studies have 17 series and 497 source slices. The source-first report
and endpoint assessments remain unchanged in this directory. The additional
review displays 51 selected cached slices: three ordered samples per series,
at approximately 25%, 50% and 75% of the series depth. The two sheets for each
case are in `cache-panels/case-NNN-II/`.

These images use the existing normalized, resized 256 × 256 uint16 cache. The
renderer maps each sample linearly to uint8 and pastes it without another
resize, rotation, flip or crop. This is a technical correspondence review, not
a diagnostic reading or a new native-resolution source review. Literal side
tags remain visible and unmodified; the renderer makes no side assignment.

`render_cache_sets.py` checks the frozen source/header evidence, all 17 actual
NPZ hashes and array shapes, and complete source SOP/hash membership before
rendering. Each panel manifest records its full source row, cache and metadata
hashes, native fingerprint reference, source geometry, ordered index and exact
display placement. `verify_cache_panels.py` separately reloads the corrected
inventory and all caches and compares every pasted display matrix exactly.
Its result, `cache-panels/display-source-integrity.json`, passed all 497 source
memberships, 17 complete cache arrays and 51 displayed matrices.

Both scripts require new output locations and preserve completed outputs.
To reproduce, copy this stage's scripts and frozen source/header evidence to a
new sibling staging directory, leaving generated `cache-panels/` out of the
copy, then run its `render_cache_sets.py` followed by `verify_cache_panels.py`
with the workspace Python environment. The referenced original corrected
inventory, cache, metadata and source evidence must remain available at their
bound workspace paths. The new run receives its own manifests; do not replace
the preserved original run.

Each `independent-cache-visual-assessment.json` was frozen after personally
viewing every displayed sample and before receiving the parent's visual
findings. Any subsequent comparison must be stored separately. No clinical
drafts, reviews or official target values were opened, and no source, cache,
annotation or runtime files were changed.
