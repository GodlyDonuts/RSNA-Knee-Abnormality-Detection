# Representative cached views: 042:21, 044:31, 045:38

The unchanged rendering protocol selects three ordered slices per complete series (25%, 50%, 75%), verifies every full NPZ hash and array contract, and binds all source identities to the corrected inventory. Display pixels are the existing 256×256 normalized uint16 cache mapped linearly to uint8 with rounding and pasted without additional crop, resize, rotation or flip.

This pass covers 16 complete cache volumes, 423 source memberships and 48 displayed matrices. It performs no new network requests or native DICOM decoding. Source tags, reports, clinical annotations and all caches remain unchanged. The source-first technical assessment and each observer freeze remain separate immutable artifacts.

To reproduce, run `render_cache_sets.py` followed by `verify_cache_panels.py` with the workspace virtual environment from a fresh copy of this stage that has no `cache-panels` output. Both scripts reject existing outputs. The renderer differs from the prior group only in the frozen technical-summary hash and case list; the verifier differs only in expected source counts.

For an independent visual pass, open only each case's `panel-manifest.json` and its two `cache-series-page-*.png` images before reading `independent-cache-visual-assessment.json`. Root visual/numeric review is pending. Publication remains on hold; these staging artifacts make no new training-readiness claim.
