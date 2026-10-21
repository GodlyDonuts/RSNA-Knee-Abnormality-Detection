# Image-context coverage probe

The trained predictive model receives fixed descriptors from every unique MRI
frame, but its CNN sees only24 uniformly selected center triplets per masked
view. The complete descriptor/support roster therefore must not be described
as complete high-resolution encoder coverage.

Read the original3,887-fitting-study cache's alias/acquisition/ordinal arrays
only on owning evc33, using the accepted index hash and original immutable
file-stat proofs. Compare exact current24-center coverage with a fixed96-center
alternative. This measures unique triplet frames and acquisitions without a
center; it does not measure diagnostic evidence or accuracy. No raw pixels,
descriptors, labels, protected cohort, model weights or predictions are read.
Only aggregate counts/quantiles leave the node. One CPU, zero GPUs, zero new
shared/node-local files, no uploads or notebook actions. The probe cannot alter
training or the already-submitted C100 version.

Four focused tests cover sampling arithmetic, acquisition/matrix-group neighbor
boundaries, alias deduplication and strict integer-only NPY parsing. No actual
cohort result is claimed until the owning-node run returns. A denser inference
candidate would still require measured model/runtime behavior and its own exact
acceptance; neither more coverage nor spare VRAM establishes better accuracy.
