# Conditional additive predictive evidence

Decision recorded October 20, 2026, before any full predictive fit or B/C/C0
competition score. This is a prepared, secondary exploratory test. It has not
run on a GPU, been submitted, or improved accuracy. The original matched
[C−B and C−C0 decision](../predictive-full-fitting-20261020-v1/evaluation-decision.md)
remains the primary experiment.

## The one test

Preserve every member and every original blend in the complete scored R6
ensemble. Add the **fixed-final C model's complete two-mask posterior**:

`final = 0.75 × midrank(original R6 final score) + 0.25 × midrank(C posterior)`

Apply ranks separately for each of the 12 targets across the complete official
test roster. Use average ranks for exact ties, normalized as `(rank − 0.5)/N`.
Do not blend raw Bayes factors into R6 odds: their dependence is not established.
Rank fusion avoids assuming that the two models' probability scales are
calibrated alike, but it does not guarantee a gain.

The conditional trigger is:

1. Actual fixed-final B, C and C0 submissions all finish with scores and without
   an execution error.
2. C scores at least **0.949** and strictly above both B and C0 at Kaggle's
   displayed precision.
3. Exact combined-source execution, output and runtime acceptance pass, and
   fresh account time, daily allowance and storage fund this additional test
   without taking the primary experiment's remaining allocation.
4. Launch before the existing **October 22, 12:00 UTC** scored-final target.

The threshold is an exploratory prioritization rule, not statistical
significance. Use exactly this weight once: no class-specific weights, alpha
grid, fitting-label calibration, automatic alternative-arm substitution or
protected-label reads. If the trigger fails, do not run this conditional test.
If it runs, success requires a displayed score above both original R6 and
standalone C; retain a tie or decline as a negative result. Public aggregate
scores cannot establish private generalization or paired confidence intervals.

The previous R6 **member replacement** scored 0.957 versus the original 0.958.
This new route retains the complete baseline and adds an independently trained
prediction mechanism. It is not a claim that rank averaging is novel.

## Exact original baseline

The [original hidden receipt](../../audits/exact-R6-entry-20261018-v1/actual-hidden-terminal.json)
binds entry **57333150**, notebook
`godlydonuts/rsna-public0958-script-b472091d7a32` **v2**, kernel **139271768**,
to actual **COMPLETE /0.958**.

| Binding | SHA256 |
| --- | --- |
| Actual scored script | `affa44363c530eed37878d083707d4fc50dcf1571d8dfcce12fd1c5a7d4bc581` |
| Prepared script with the same Python AST/cell strings | `b472091d7a323e0165958ca01e6f023b21db9c3d2b10d3f79ba8962fb83efc5b` |
| All eight original code cells | `300b34417bc1bc91018a4bb50fa9440403e904ab7abcc7236e432e61932df56f` |
| Original notebook | `fac1723c32de2d27f59e10ebca7102327bb33346117a3fec3a58937ecf2d3a0e` |
| R6 v1 manifest | `6508ce6921eda986c885b2226fa55dd4d2aa8e5fc570b4c007c9a60c05bfc648` |
| Original public CSV | `5e9ba665f3e30642e18dbb555eb7cbd16ab8a659ffb5f7226c72aaac44c9df1b` |

Retain original versioned inputs:

- `heliosli/rsna-knee-r6-frozen-fe055406` **v1**, never the different R7 v2.
- `nartaa/rsna-knee-publication-swa-weights-20261007` v1, Dataset12431568.
- `goodpjw2008/rsna-knee-2-5d-convnext-reader` v1.
- `heliosli/rsna-orthofoundation-fold0-epoch7-20261004` v1.
- `goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958`,
  original kernel-version359009299.
- Competition test data plus the actually preserved/qualified C Dataset.

Original Dataset-version IDs are 20297367, 20325538, 20440357 and 20611930.
Resolve each exact original attachment through its existing receipt when
preparing the candidate; do not infer a current version from a slug.
Third-party code bodies, weights and predictions remain private. The compiler
must consume the exact original driver rather than the similarly named paired
replacement source.

## Execution order matters

R6 cell 4 installs timm1.0.25 and pydicom3.0.1 globally. The predictive T4 probe
ran timm1.0.29 and pydicom3.0.2. A single Python namespace, or running R6 first,
would silently change predictive inference.

The prepared route therefore runs:

1. The exact qualified C competition source in a fresh child process.
2. After that child exits, the complete original R6 driver in another child.
3. Strict UID-aligned rank fusion of both complete final CSVs.

The original R6 driver retains its IPython cell execution, all eight cells,
credits, every model, preprocessing and blend. Do not share transformed tensors
or encoder features: the models have different trained weights and different
pixel normalization, slice selection and masking. Only unchanged input mounts,
the test roster and the operating system's page cache can be reused directly.

An outer total-duration limit must cover both process trees, including startup,
preparation, inference and output. R6's internal 8.3-hour clock starts too late
to cover preceding C inference. A child error, incomplete roster, fallback,
stale file, runtime mismatch or timeout must not publish a partial submission.
One exact candidate version produces the complete fused `submission.csv` during
its public run. Only after that actual run passes output and runtime review may
the canonical coordinator submit **the same immutable code version**. Do not
upload the three-row CSV, embed the new acceptance into the source, or run the
same full pipeline again merely to change a mode. Creating the output is not
authority to submit it.

## What remains unmeasured

The original R6 public execution took **475.230 seconds**, completed all eight
cells and reported zero fallbacks. Its hidden run scored, but its exact hidden
elapsed time and peak VRAM have not been established. Separate successful runs
do not prove that their sum fits nine hours.

The new C trained qualification and standalone public/hidden results do not
exist yet. The fresh-head probe and cluster rendering speedup cannot replace
them. Sequential model processes avoid adding their GPU-resident allocations,
but a combined actual run must still measure memory, CPU preparation and total
wall time on the exact two-T4 runtime.

Before any competition entry, require the original R6 public CSV to match its
recorded bytes and C's public CSV to match the actual standalone accepted C
source. Record complete stage timings and review a combined hidden-population
projection for at least the official approximate **1,300 studies**, with at
least **600 seconds startup and 1,800 seconds tail margin**. The sum of both
component projections and those margins must fit nine hours. Exact evidence
hashes must support each component projection; the compiler does not infer them
from the public elapsed time. Public-three-study equality establishes
integration, not accuracy or hidden-tail coverage.

The current **3,360 GPU-VM-second primary reservation does not fund this extra
run**. No additional Slurm job, Dataset, notebook, submission, download, storage
liability or private mirror is admitted by this document.

## Local implementation

- `fusion.py` performs the fixed rank calculation and strict full-roster/schema
  checks without labels, model loading or filesystem access.
- `compile.py` calls the existing final C compiler using actual final export,
  trained qualification and standalone public acceptance. It embeds the exact
  original R6 driver without modifying its body. Generated source contains
  private third-party bodies and must remain outside Git.
- `supervisor.py` runs C and R6 sequentially in separate process groups,
  checks both original acceptance receipts and runtime bindings, and records
  stage times. A timeout or failure removes partial final output.
- The compiler module's separate pure `accept_entry` gate reviews the actual
  combined public receipt and an evidence-bound runtime decision **after**
  execution. It creates no submission. The canonical coordinator must additionally
  bind the same kernel/version/source, exact input versions, Docker image and
  independently retrieved complete fused CSV, with fresh account allowance and
  resource funding. No generated source is changed after this review.

The focused tests cover tie ranking, UID alignment, monotone transformations,
invalid inputs, process ordering, descendant cleanup, stale-output rejection,
exact source/public-output binding, and runtime-admission failures. These are
engineering checks, not clinical results. No final model exists with which to
compile or execute this combined candidate yet.
