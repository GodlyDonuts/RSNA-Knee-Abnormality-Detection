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
