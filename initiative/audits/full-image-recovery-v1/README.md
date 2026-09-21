# Completed image recovery and audit

The exact two official Kaggle DICOM encodings are restored in private HF source revision `0b5e410288232f570c69ba4f44723356e6ae6ab7`. The baseline source, inventory, cache and strict decoder remain preserved.

All 819,635 slices in 24,386 series passed the complete effective-cache audit with zero cache warnings. One cross-study nonconstant pixel edge is retained for leakage grouping; there is no overlap with the three released test studies. This does not establish hidden-test isolation or patient identity.

The training selection preserves all 4,407 studies and 24,338 of 24,371 series, excluding 33 fully equivalent model slots under the reviewed exact-content, geometry and feature policy. All 15 released test series are retained. Original inputs and every file remain unchanged. Six accepted anatomical-plane overrides remain in the selected training table.

See `preservation.json`, the complete audit, selection plans, per-series lineage and root verification for exact hashes, external inventory location and rebuild instructions. The 82.6 GB cache is locally preserved and reproducible from pinned sources. Training readiness remains false pending full report/semantic/correspondence/leakage and final smoke gates.
