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

## Separate visible-only contrast candidate

`rsna.competitive.predictive.visible_contrast` implements a separate experiment.
For each complementary mask independently, it estimates acquisition-specific
1st/99th percentiles from at most4,096 deterministic visible positions per
canonical frame. Alias copies receive no extra weight. The span floor0.01
bounds amplification at100; hidden pixels are refilled with0.5 before rendering.
Original64D targets and complete support priors remain unchanged. Original
offset and log-span are appended to token/region metadata so the evidence
predictor retains intensity-scale information.

The matched control retains original images and exposes the same two new
metadata coordinates. Both need a separately initialized10-coordinate graph;
neither is compatible with an ordinary8-coordinate checkpoint. Existing
B/C/C0, queued C100, cache workers and serving sources do not call this module.
No variant has GPU training admission, a checkpoint or an accuracy result.

`run_candidate_once.py`/`probe_candidate.py` prepare one original-input CPU
experiment on exactly the same12 fitting cases, excluding four engineering
cases. It compares exact unmasked contrast/clipping, requires unchanged
targets/supports and original-control images, replays the prior aggregate
measurement, and changes hidden pixels in one real fitting study to test
input/scale invariance. It uses one CPU/zero GPUs/12GiB for at most10minutes,
streams the candidate source through stdin, and writes no remote files.
Local tests cover source intervention, aliases including alias-only
acquisitions, constant images, target preservation and10-coordinate backward.
Actual CPU **984583 COMPLETED0:0 /2:49** from source17788060; see
[original-input result](actual-candidate-input-result.json) and
[terminal](actual-candidate-terminal.json). It reused2,835 frames and576 windows
per variant, reproduced the original measurements, and found:

| Exact visible statistic | Original | Corrected |
| --- | ---: | ---: |
| Median study1st–99th percentile span |0.16117|0.97692|
| Minimum study span |0.01517|0.95688|
| Studies below0.05 span |3/12|0/12|
| Median pure-visible coverage |39.29%|39.29%|

All original descriptor targets, observation masks, complete support priors and
first eight metadata coordinates stayed identical. The ten-coordinate control
has exactly the original image tensors. On one original fitting study, replacing
all pattern0 hidden pixels with1−original changed the target descriptors while
leaving pattern0 images, metadata and normalization statistics bitwise identical.
The other complementary context is correctly permitted to change.

The normalized rendered pixels' median fractions at0/1 are12.54%/0.62%.
Zeros include anatomical background and removed encoding offsets; these are not
measurements of newly discarded tissue information. The experiment does not
establish that percentile clipping preserves every lesion. Normalized rendering
also costs59.70seconds versus27.95seconds for the original across12studies,
excluding raw-cache loading; cached training can amortize preparation, but this
is an additional serving-time cost to qualify.

Decision: the isolated correction passes this real-input engineering test and
is a viable candidate for a matched learning experiment. Compare normalized
images against the ten-coordinate unchanged-image control from one common
initialization, including the full scheduled LR peak. Do not infer accuracy from
expanded dynamic range or tiny fitting losses. This receipt does not fund or
launch another GPU, admit a full variant or alter the healthy primary workers.
A separate bounded training run, recovery acceptance and fresh resource review
remain necessary before full fitting or competition deployment.
