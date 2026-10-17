# Connected witness paths: a testable next MRI mechanism

The working prototype aggregates a **latent, variable-length spatial path**
through ordered MRI windows. A single-window path is allowed: a small focal
finding need not appear throughout an acquisition. It differs from taking a
mean or maximum over independently computed pairwise similarities. The current
aligned/null full-reader runs are a separate experiment and remain unchanged.

This builds on [structured attention](https://arxiv.org/abs/1702.00887) and
[attention-based multiple-instance learning](https://proceedings.mlr.press/v80/ilse18a.html).
It is not a claim that latent paths, MIL or structured attention are new. The
falsifiable question is whether connected, source-qualified evidence adds useful
ranking information to this particular MRI pipeline.

## What is actually implemented

`prototype.py` computes the exact log partition of a mixture of possible starts,
spatial token paths and stopping positions using a backward dynamic program.
Start tokens follow the stationary weighted spatial measure. Continuing across
an edge has prior probability
`sigmoid(kappa[target] - log(gap_mm / reference_gap_mm))`. Every earlier position
may stop, and the final position must stop. Zero evidence integrates to prior
mass one, with an explicit numerical zero baseline subtracted. Logits are not
clamped. The 9.900010275666745mm reference is the actual fitting-metadata median
of dense, source-disjoint repeat gaps; it is a unit, **not** a lesion-size cutoff.
An arbitrary reference-unit change can be absorbed by the learnable kappa.

The control replaces a spatial joint Pi with `a outer a`, preserving the same
stationary weighted token marginal while removing its step-to-step dependence.
Image evidence, possible starts, stopping prior and physical gaps are shared.
The prototype also mixes chains by their number of possible starts; overlapping
phases are not treated as independent confirmations.

`check_math.py` actually passed exhaustive path enumeration for a tiny case,
double-precision gradients, finite extreme-evidence/continuation checks, exact
zero-evidence baselines, coordinate-permutation invariance and six invalid-input
refusals. In a toy equal-histogram/equal-adjacent-dot-summary example, coherent
versus switching evidence scores 2.6550 versus 1.9086 with the physical prior,
and 2.3499 versus 2.3499 with the control. These are **synthetic arbitrary scores**,
not MRI predictions, a collision of the trained CNN, or clinical accuracy.
[Actual result and source hash](mathematical-checks.json).

The current mathematical implementation accepts a **fixed reversible joint**
with positive token mass. Actual source geometry can yield directed or changing
in-plane displacements. It must not be silently symmetrized, dropped or claimed
accepted by this prototype. This layer has no MRI loader, evidence head,
optimizer, checkpoint, serving runner or measured hardware cost. It has not been
staged on Newton or used to train a model.

`source_chains.py` now implements explicit metadata qualification: partition
offset3 edges without crossing blocked source gaps, prove disjoint original
slice support within each chain, retain focal singletons and account for every
window. Changing/directed/nonstationary priors are counted as unsupported, never
symmetrized or repaired. Synthetic source partition, gap/focal, unsupported-motion,
missing-operator and four invalid-source checks passed. The one-time read-only
32-fixture CPU assessor is prepared; it loads only original metadata and executes
the new small modules in RAM, with zero new node/shared files or MRI/target reads.
Its original intent/actual receipt, rather than this plan, determines execution.

## Actual first input result and a necessary generalization

CPU951433 actually assessed the original32 engineering metadata fixtures without
reading pixels, maps or clinical values. Every2345 original window was accounted
for, but the fixed-joint model fully supported only13/16 fitting and11/16 outer
fixtures.53chains/250windows had changing edge kernels. Their changes may include
header-precision effects; this does not establish clinically meaningful motion.
[Original result](actual-source-geometry.json). Do not drop those8studies or
silently symmetrize their geometry to make this prototype pass.

`variable_paths.py` therefore implements the exact partition with **a separate
original transition for every edge**, without requiring reversibility or a
stationary measure. Start priors are explicit; the new source builder uses a
uniform prior over every observed token at each possible start. Original
nonnegative kernels are normalized by outgoing mass, without symmetrization.
Zero-outgoing observation states remain unsupported until an explicit policy
exists. This is a different prior construction from the first fixed prototype,
shared identically between the two new arms.

The control is derived conditionally on each possible start: propagate the
physical prior to obtain `m[start,time,token]`, then independently draw tokens
from those same measures. Thus **every single-position prior marginal given each
start is preserved**; evidence-dependent path coupling is removed. The shared
class-specific stopping prior is independent of token state. This is a stronger
statement than replacing changing matrices by an arbitrary shuffle, and says
nothing about equality of evidence-conditioned posteriors or predictions.

Exhaustive tiny physical/control path enumeration, enumeration of every
start-conditioned prefix marginal, both unary/continuation gradient checks,
focal/FP32/extreme-prior cases and five source-prior refusals passed. The old
stationary model is reproduced as an exact special case of both mathematical
arms. [Actual mathematical result](variable-mathematical-checks.json).
Synthetic source checks retain changing/directed kernels and refuse missing
outgoing states. This reference implementation has **quadratic sequence-length
cost/storage** for the conditional null; dense-MRI throughput is unmeasured.

**Actual CPU951534:** the varying-kernel layer now supports all32 original
fixtures/all2345 windows, including all53chains the first prototype could not
represent. All1841 source-disjoint repeat edges are retained. Actual zero-evidence
and conditional-prior-mass errors were at most1.1921e-7 in FP32; execution took4.19s.
No pixel/map/target/model values were read and no node/shared files were created.
[Exact original assessment](actual-variable-source-geometry.json). None of these
32 fixtures had an individually nonreversible kernel at tolerance1e-6: support
for directed kernels is established synthetically, not claimed as a measured
phenomenon in these particular acquisitions.

## Learned reader and bounded next execution

`path_reader.py` implements the learned1024-to64 projection,12 evidence channels,
class-specific continuation parameters, common full-map image/nuisance features
and zero-start **unbounded** residual. It has68560 parameters. Physical versus
independent arms use identical maps, starts/gaps, summaries and architecture;
only path coupling changes. Overlapping phases/acquisitions enter a mixture of
possible starts, not multiplied confirmations. Encoder weights remain external
and frozen. The unchanged canonical `SpatialTrainer` schedule/state/RNG loop is
reused explicitly with a new reader and fresh AdamW; no original module changes.

Fresh-process synthetic CPU full/stop-after1/resume checks actually passed for
both arms: complete learned state/AdamW/LR/order/cursors/all-RNG/history and each
epoch agree exactly, as do16x12 prediction CSV bytes. Both arms begin at the
same model. [Actual result](synthetic-reader-recovery.json). These synthetic
fixtures use8channels/width4 and do not accept full1024/64 real-MRI GPU execution,
an encoder, diagnostic performance or a production candidate.

The following private transfer was prepared but **retired before intent or any
remote execution** when the original evc40 jobs finished; all inputs are already
resident there. No copy, deletion or GPU cancellation occurred. Original plan:
one private transfer of the **already whole-byte verified32 cached
feature cases** from evc33 to evc26, through allocated CPU endpoints and a pipe
on Newton. No feature arrays/rosters cross to the Mac/Git/HF. Original source
files stay intact. New evc26 root is capped3GiB, including all copied data and
future bounded engineering controls/checkpoints, with20GiB node reserve and zero
new shared data/files. This does not fund full native-cache or production fits.
The exact sender/receiver logic passed a synthetic positive case and10negative
hash/inventory/path/size/link/missing/duplicate/cap/overwrite checks, including
owner-only permissions. [Actual transport checks](fixture-transport-checks.json).
Existing staging intent/root forbids repeating a copy; reconcile read-only.

Only after actual fixture and source/input/runtime admission, run one bounded
H100 full/stop/resume recovery on16 fitting plus16 prediction-only original
cases,2epochs/4updates. This is engineering, not a model to submit or an accuracy
experiment. Actual CPU951824 has now admitted all32 full1024x12x12 cached inputs, source
bindings and zero-start parent equality. Distinct1H100/twoCPU recovery951853
was submitted once on permitted physical2-H100 evc40 and is PENDING(Priority).
It is a bounded4-update engineering check, not a production accuracy fit.
[Exact original dispatch](actual-real-GPU-submitted.json). The completed original
paired reader jobs and their final analysis remain unchanged.

The original paired experiment is now complete: internal aligned0.9222304,
null0.9211719, parent0.9205878 on1156 weak-label studies. Geometry delta+0.0010585
has95%[-0.0005818,0.0030275], with1186/4000macro draws undefined. This is not a
robust macro gain or a competition result. Preserve the negative/uncertain result.

Reuse evc40's original32 full-map inputs for the new bounded recovery instead.
Its64MiB engineering cap is reassigned within the completed full-run320MiB:
retain256MiB for every original artifact, keep total reader512MiB/node3GiB and
the original node40 floor612022943803 unchanged. No new large data/weight copy or
shared files. The CPU stage must verify all32 actual1024x12x12 inputs, original
16 fitting label/mask hash and target-free outer provider, plus zero-start parent
logit equality on actual fitting/outer examples. Only an explicit agent decision
bound to that actual CPU receipt may dispatch one1H100/2CPU/15minute job on
physical2-H100 evc40. The original shared trainer loop, fresh-process recovery
and private sixteen-case CSV equality must pass for both new path arms. This
does not qualify a dense/online provider, train an encoder or establish accuracy.

## Why test this next

The [actual full fitting census](../../audits/spatial-throughplane-information-20261017-v1/actual-coverage.json)
uses all2247 fitting studies, within the currently selected at-most6 series:
16-slice selection retains185929 of386846 original source slices, or48.06%.
Dense adjacent triplets would create363568 windows versus162651 now (2.235x).
Those dense triplets also change channel context; this is not just presenting
more copies of the same input. Full sparse-plus-dense source-support union is
513788 triplets (3.159x the current count); matching source tuples alone does
not establish equal normalized input tensors or accepted encoder reuse.

Current source-disjoint repeat edges span a median20.0001mm. Dense raw triplet
repeats span a median9.9000mm. Both retain every fitting study. The original graph
already has linear-sized offset1/offset3 relations and blocks unusually large
native gaps. The [spacing probe](../../audits/spatial-throughplane-information-20261017-v1/actual-result.json)
shows absolute-normal-spacing invariance for fixed maps, not the absence of all
longitudinal topology or proof that gap sensitivity improves disease ranking.

## Proposed next comparison; not a dispatched campaign

Prepare a matched2x2: original sparse versus dense raw triplets, crossed with
physical versus factorized latent paths. A shared frozen encoder can supply the
four compact readers, but only after exact input/forward equivalence and
recovery have been demonstrated. Share initial state, accepted report labels,
masks, study order, optimizer budget and stopping. Preserve the original sparse
N3 prediction as an additional reference. Do not use validation to choose an
epoch, target subset, stopping prior or geometry threshold.

Within each acquisition, partition overlapping consecutive triplets into
source-disjoint offset3 chains, splitting at disallowed native gaps. Prove
original support/chronology and address changing/directed geometry explicitly.
Do not connect acquisitions using unqualified registration. Nominal CNN token
locations are not segmentations or localized receptive fields. Bag-level labels
must not become positive labels for every MRI slice.

Before a new GPU fit: qualify the actual input/chains on the original bounded
fixture; accept exact real full/stop/resume states and predictions; measure
end-to-end time/memory on an allowed physical1/2-H100 node; fund retained bytes
and publications. The hypothetical fitting-only dense full-map cache is199.71GiB
and is **not** funded. Stream source images/features instead. The existing
three-epoch paired experiment, its final1156-study readout and its two healthy
H100 allocations retain their exact original recipe.

Promotion requires fixed-final improvement over both the appropriate parent
and matched control, all12 targets with paired whole-component uncertainty,
then replication on a matched seed/fold and actual offline T4 runtime acceptance.
There is no measured MRI, clinical or Kaggle gain from this prototype.

## Streamed dense inputs: implementation and next actual check

`dense_inputs.py` now implements sparse16 and every-source-slice triplets with
the original uint16 normalization and grayscale resize, in bounded12-window
chunks. No dense image/map cache is created. Source supports, chronology and
all image windows are retained; the unchanged native-gap rule blocks physical
path edges without dropping images. Within each selected series, dense coverage
also changes RGB channel context. Earlier `compact_source_neighbors` work only
changed donors at the same original selected centers; this new path expands
centers as well and is not claimed as a novel adjacent-triplet idea.

162 synthetic tensor-reference/chunk checks passed at multiple counts and
resize sizes, including short/focal inputs and sparse/dense equality below17
slices, plus native-gap/path accounting and six invalid-source/input refusals.
[Exact result](synthetic-dense-inputs.json). These are not original MRI acceptance.
The distinct `assess_dense_once.py.txt` coordinator is prepared for one CPU
RAM-only audit on the32 original engineering studies: metadata/cache hashes,
exact original sparse graphs, complete independent normalized-slice tensor
reference and dense varying-path accounting. It does not read clinical targets,
instantiate an encoder or fund dense model training. Original intent/receipt
determines whether it actually ran.

**Actual CPU951889 completed:** all32 original native MRI studies/168series/5234
source slices were read from their owning-node cache with metadata/cache hash
checks. Sparse2345 and dense4898 windows were tensor-equal to the independently
normalized complete-slice reference; every sparse graph equalled its original
accepted graph. All4898 dense windows/4394 repeat edges are supported by the
variable-path layer, with zero dropped/unsupported windows.34.85seconds, oneCPU,
zero GPU, zero new node/shared files, no clinical target reads.
[Original pixel/tensor/geometry receipt](actual-dense-inputs.json). This does not
accept encoder output equality, online GPU recovery, dense throughput or accuracy.

`observe_real_gpu.py.txt` is a finite read-only follower for original951853.
It may collect the exact source-bound engineering receipt using oneCPU only after
actual0:0 completion. It has no GPU retry or production dispatch authority.

## Online frozen-encoder integration

`online_inputs.py` adds streamed sparse/dense full-map providers. Encoder chunks
remain global12-window batches across acquisition boundaries, matching the
original sparse encoder call order. Both samplings retain the **same original
sparse16 parent teacher**, rather than changing the teacher when coverage changes.
Native `[0,1]` grayscale tensors receive the original final `2*x-1` transform
before encoding; a synthetic end-to-end reference test caught its omission in
the new wrapper, and it was fixed before any GPU dispatch or training. This is
not an explanation of old Kaggle failures or a change to old numerical sources.

Synthetic encoder integration now reproduces original sparse full maps, source
plan and parent logits exactly, keeps dense teacher logits equal and retains all
dense windows. Every frozen parent parameter/buffer stays byte equal; train-mode
parents and metadata drift are refused. [Actual synthetic result](synthetic-online-inputs.json).
This small synthetic encoder cannot accept the real72.9M-parameter N3 encoder.

The distinct `dense_gpu_once.py.txt` prepares one1H100/twoCPU/15minute **input
probe**, on physical2-H100 evc33, with no optimizer or clinical label reads. It
checks real sparse full-map/parent equality against all32 original cached cases,
real dense full-map coverage with the same teacher, zero-start readout on a
fitting and outer case, exact frozen state and measured time/peak memory.
Source/runtime/metadata/funding are checked by oneCPU before GPU submission.
It reassigns32MiB inside node33's original full320MiB allowance, retaining288MiB
for every original/future artifact, with no new shared bytes or large copies.
It does not duplicate the cached-reader recovery951853 or two-device benchmark
950940. Original intent and actual submission/terminal receipt decide execution.

**Actual CPU951962 completed0:0:** exact sources/runtime, physical2-H100 node,
original118-module binding and reserves admitted. It submitted independent
no-optimizer input probe **951964 ONCE** on evc33,1H100/twoCPU/15minutes,
sourceaf6f439a. Latest actual state PENDING(Priority). Original951853 and950940
remain queued and unchanged. [Original dispatch](actual-dense-GPU-submitted.json).
The finite read-only `observe_dense_gpu.py.txt` can collect its source-bound
receipt after actual0:0; it has no GPU retries/production authority.
