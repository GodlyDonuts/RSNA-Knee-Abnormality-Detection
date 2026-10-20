# Visible MRI contrast under fixed encoding normalization

Prepared October20 while the original full-cohort cache jobs run. This is a
bounded, read-only CPU measurement on the same twelve fitting engineering
studies, excluding the four evaluation cases. No model, clinical label, changed
normalizer or new cache is involved. No actual result exists until the original
CPU receipt is collected.

The proposed mechanism requires removing information from source pixels before
rendering. Current scaling uses DICOM encoding bounds; half-filled regions use
0.5. A valid concern is that some scanners occupy a narrow fraction of those
bounds, making the occlusion much stronger than the retained MRI contrast.
This is a risk to measure, not an assumed root cause or a reason to disrupt the
frozen B/C/C0 experiment.

The collector reads exact existing FP32 rendered images and descriptor DCs,
bound to original input proofs. It reports equal-context/equal-study summaries
of non-0.5 pixel quantile spans, standard deviations, exact0.5 fill fractions and
near-boundary fractions. Non-fill pixels include bilinear interpolation borders;
they are explicitly a proxy, not an anatomical tissue mask. The predeclared
span<0.1 count is descriptive, not a clinical pass/fail threshold. Per-study
records, MRI pixels and identifiers remain on evc27. The observation creates no
remote files and requests one CPU, zero GPUs, 2GiB RAM and at most five minutes.

If spans are healthy, do not pursue contrast repair on this evidence. If many
are compressed, inspect whether normalization computed exclusively from visible
source pixels could restore the encoder input range without exposing withheld
pixels. That would require a separate numerical/input experiment; no present
cache or live source would change, and it would not itself establish accuracy.
