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
boundaries, alias deduplication and strict integer-only NPY parsing. Original CPU1001376 actually completed0:0 in5.54seconds. It read21.78MB
of integer metadata across all3,887 fitting studies, with zero pixel/descriptor/
label reads and no new remote files. The [actual receipt](actual-cohort-coverage.json)
shows study-mean unique triplet coverage43.37% at24 centers versus97.32% at96;
median42.86% versus100%. Four studies missed five acquisitions at24; none did
at96. This is image-encoder coverage, not a change in the already-complete
descriptor/support roster. The [decision](agent-decision.json) selects fixed96
for a bounded same-model inference test; no96 notebook or entry exists yet. A denser inference
candidate would still require measured model/runtime behavior and its own exact
acceptance; neither more coverage nor spare VRAM establishes better accuracy.
