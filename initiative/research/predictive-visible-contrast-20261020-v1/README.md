# Visible MRI contrast under fixed encoding normalization

Measured October20 while the original full-cohort cache jobs run. This is a
bounded, read-only CPU measurement on the same twelve fitting engineering
studies, excluding the four evaluation cases. No model, clinical label, changed
normalizer or new cache is involved. Original CPU984478 completed0:0 in4seconds;
the exact visible-pixel follow-up984498 completed0:0 in53seconds.

## Actual finding and decision

The [first rendered-input measurement](actual-result.json) found3/12 studies
with a non-fill1st-to99th-percentile span below0.1. Its median span was0.3288,
but bilinear mixtures with the0.5 fill can inflate that estimate.

The [exact follow-up](actual-exact-visible-result.json) reused2,835 original
frames and576 rendered windows across the same12 fitting studies. A diagnostic
hook returned the original renderer's tensors unchanged while admitting only
pixels whose bilinear hidden-source indicator was exactly zero. It found:

- Median visible1st-to99th-percentile span **0.1612**, range0.01517–0.30070.
- **4/12** studies below0.1; **3/12 below0.05**.
- Median pure-visible fraction **39.29%**, range33.24–42.42%; the complement
  includes both deliberate mask fill and interpolation borders.

This establishes substantial encoding-range compression in a subset of these
inputs. It does not show that tissue information is absent, estimate its
frequency in3,887 studies, establish clinical accuracy, or explain earlier
0.948 entries, which used a different input pipeline. Visible pixels still
include anatomical background. No clinical labels, model, held-out inputs,
per-study outputs or pixels were exported or changed.

Decision: retain the frozen B/C/C0 comparison and C100 pilot. Before creating
another data-dependent contrast branch, use normalization statistics drawn
exclusively from visible source pixels, never target pixels. Any such candidate
must preserve hidden-pixel intervention invariance and explicitly account for
the absolute scale needed by the conditional density predictor; changing only
encoder contrast may discard scale information needed to predict the original
targets. Compare a distinct candidate with the existing inputs rather than
silently changing the running study. This finding gives a concrete follow-up
to the measured weak diagnosis-specific gradient; it is not a promoted model.

The proposed mechanism requires removing information from source pixels before
rendering. Current scaling uses DICOM encoding bounds; half-filled regions use
0.5. A valid concern is that some scanners occupy a narrow fraction of those
bounds, making the occlusion much stronger than the retained MRI contrast.
This is a risk to measure, not an assumed root cause or a reason to disrupt the
frozen B/C/C0 experiment.

The first collector reads exact existing FP32 rendered images and descriptor DCs,
bound to original input proofs. It reports equal-context/equal-study summaries
of non-0.5 pixel quantile spans, standard deviations, exact0.5 fill fractions and
near-boundary fractions. Non-fill pixels include bilinear interpolation borders;
they are explicitly a proxy, not an anatomical tissue mask. The predeclared
span<0.1 count is descriptive, not a clinical pass/fail threshold. Per-study
records, MRI pixels and identifiers remain on evc27. The observation creates no
remote files and requests one CPU, zero GPUs, 2GiB RAM and at most five minutes.

`visible_pixels.py` implements the subsequent source-mask measurement. Its
separate one-time runner uses one CPU/zero GPUs,6GiB RAM and a ten-minute cap;
actual runtime was under one minute. It writes no remote file and never
reacquires source data. The interpolation-boundary regression shows that gray
mask edges cannot inflate its pure-visible contrast statistic.
