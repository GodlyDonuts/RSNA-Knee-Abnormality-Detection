# Bounded representative cache review

This extends the preserved source/header-only assessment with three ordered
representative cached slices from every series: 15 images for 016:06, 18 for
021:01 and 15 for 022:40. Each case has two sheets and a bound panel manifest
under `cache-panels/case-NNN-II/`. The earlier source/header handoff remains
unchanged and accurately describes the scope at that stage.

The 48 selected samples use existing normalized 256 × 256 cache images, not
native-resolution DICOM data. They are linearly mapped from uint16 to uint8
and pasted without another resize, flip, rotation or crop. All literal side
tags are preserved. Rendering does not infer clinical findings or named sides.

`render_cache_sets.py` verifies frozen metadata, all 16 actual NPZ hashes and
arrays, and every source SOP/hash membership before rendering.
`verify_cache_panels.py` separately reloads the corrected inventory, all 16
caches and their metadata, and compares all 48 pasted display matrices exactly.
The replay result is `cache-panels/display-source-integrity.json`: 409 complete
source memberships, 16 cache arrays and 48 displayed matrices passed. No new
network requests or native source decodes were performed.

Each panel records its complete source row, cache and metadata hashes, ordered
slice index, native fingerprint reference, source geometry, native matrix
dimensions and exact display location. Each independent visual assessment was
frozen after personally viewing all assigned panels and before the parent's
visual findings. Subsequent joint comparison must use a separate artifact.

To reproduce, copy the stage's frozen source/header evidence and scripts to a
new sibling staging directory without `cache-panels/`. Run that copy's
`render_cache_sets.py` followed by `verify_cache_panels.py` in the workspace
Python environment. Bound original cache, inventory and source evidence paths
must remain available. Both scripts require new output files and directories;
the existing completed run must not be replaced.
