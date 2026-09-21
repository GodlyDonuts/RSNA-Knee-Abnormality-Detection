# Source and header assessment of three mixed-tag studies

The assessment covers source cases 016:06, 021:01 and 022:40: 16 series,
409 source slices and 32 existing lexical endpoint headers. Each original
report was personally read and preserved before case-specific metadata was
opened. All three reports exactly match both the fresh official and archived
source CSV copies. They contain single unsided knee narratives; no explicit
bilateral report or separate second report block is present. Two reports are
brief but have complete technique/results/impression structure, so no missing
content is reconstructed.

The metadata supports three candidates with conflicting tags within one
protocol. Case 016:06 has opposite L/R tags on its paired coronal acquisitions;
021:01 has R sagittal/axial/oblique tags and L coronal tags; 022:40 has R
sagittal/axial, L coronal and blank second-coronal tags. Paired coronal physical
field centers agree despite different matrix resolutions in 016:06 and
022:40. No separate full second-knee acquisition set is established.

`source-header-assessment-and-review-plan.json` preserves exact findings,
membership, source hashes and proposed representative cache review. It does
not assign named anatomical sides or infer clinical targets. No image arrays
were opened, and no native files or headers were downloaded during this task.

`collect_technical.py` verifies the corrected inventory, existing endpoint
survey, recovery lineage and full-audit cache metadata, including every source
SOP/hash and path membership. `compare_geometry.py` checks all 32 endpoint
positions, orientations, pixel spacings and native dimensions against their
corresponding cache entries. `verify_image_centers.py` reproduces all 16
recorded image-field centers, accounting for native matrix size and spacing.
These centers are not anatomical landmarks or proof of a common registered
frame across distinct anonymized frame identifiers.

All outputs are preserved. Reproduction should use a new sibling staging
directory with the source-only preread and scripts; run the collector and
geometry comparison there rather than overwriting this completed stage.
Representative cached images are the next bounded check if authorized;
full-native escalation is reserved for contradictory anatomy or geometry.
