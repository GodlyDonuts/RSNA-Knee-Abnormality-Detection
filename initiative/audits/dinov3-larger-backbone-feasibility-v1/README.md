# Larger DINOv3: read-only feasibility review

The owner asked about a larger backbone and whether the MRI input feed is
optimal. This review nominates **DINOv3-H+/16** as the next larger-backbone
hypothesis. It does not change the official accepted L base, active source-cache
jobs, training bindings or frozen supervision.

Meta's [official model card](https://github.com/facebookresearch/dinov3/blob/main/MODEL_CARD.md)
lists L at300M, H+ at840M and7B at6,716M parameters. Their general vision
benchmarks do not improve monotonically with size on every task; they provide
no knee-MRI transfer guarantee. H+ is2.8times the base parameter count of L;
7B is approximately22times. These ratios do not predict MRI speed or accuracy.

Authenticated metadata/config and weight-HEAD reads succeeded for the pinned
H+ and7B repositories. No weights were downloaded and no license application,
training job or competition submission was made. `official-model-metadata.json`
records exact source revisions, config contents/hashes and official file
sizes/LFS SHA256 metadata; `weight-head-access.json` records the readable H+
weight object and first7B shard. A HEAD or metadata check is not full-byte weight
verification, and the remaining7B shards have not had an authenticated HEAD test.

| Model | Exact parameters in local meta graph | Official FP32 payload |
| --- | ---: | ---: |
| H+/16 | 840,592,640 | 3,362,432,800bytes |
|7B/16 |6,716,035,072 |26,864,210,088bytes over six shards |

`verify-meta-graph.py.txt` constructs the graphs on the meta device using pinned
Transformers5.6.1 and reviews adapter projection names without allocating model
weights. Both have SwiGLU gate/up/down projections; L has an ordinary GELU MLP.
The existing rank32 policy targets152 H+ projections/15,400,960 parameters or
1847B projections/51,380,224 parameters, excluding each SwiGLU gate projection.
That exclusion is explicit, not a claim that all gated FFN projections are
adapted. Extending to the gate projections is a separate policy decision.

Our current `rsna.competitive.backbones` admits only the approved L/B contracts,
and its official namespace migration is bound to the exact L source. H+ cannot
be enabled by editing its model name alone: architecture semantics, head input
width, strict state-key migration/shape loading, provenance, adapter policy,
delta reconstruction and role-bound publishers/deployment need reviewed support.
Nothing in this audit implements or accepts that support.

H+ is the preferred first scaling experiment because its memory/payload increase
is materially smaller than7B. That is an engineering judgment, not a measured
MRI advantage. The current offline route keeps FP32 base weights, so the7B
base alone exceeds a single16GiB T4; BF16/quantization/sharding would require a
new accepted loading/inference route and would not establish runtime feasibility.
Newton H100 capacity does not establish Kaggle hidden-rerun capacity. Require a
real H100 recovery/throughput pilot and representative offline T4 inference
before promotion. Preserve20GiB/5,000files after future obligations: present
shared quota has approximately20.25GiB funded reserve, insufficient to stage a
new3.36GB shared H+ payload while retaining the reserve without fresh storage
planning. No large local or cluster transfer has been attempted.

Compare L and H+ on identical accepted optimization/development membership,
resolution, slice/series feed, fixed-final endpoint, effective batch and head
type. Keep rank32 initially, documenting its larger absolute adapter count on
the wider/deeper backbone. The head input projection width must also change.
This answers a backbone/representation question, not a pure parameter-count
causal estimate. Hold native resolution and slice-context changes separate.

## Current MRI feed and the next input hypotheses

The sampled L run feeds up to six selected series, up to sixteen evenly spaced
slices each,224pixel input from the256cache, independent2D DINO slice embeddings,
and target-specific slice/series attention using plane/sequence/position features.
The loss is on the final **study prediction** with unknown findings masked; it
does not assert that every slice shows each positive abnormality. This is a
valid study-level multiple-instance formulation, not a demonstrated optimum.

`augment=false` is a consequential limitation: `_indices` returns the same
uniform locations each epoch and series selection is stable. More epochs repeat
those images; they do not expose the omitted source slices. There is also no
neighboring-slice context in these completed adapter runs. The existing
`augment=true` changes both series selection and stratified slice positions;
using it directly would confound the two. A stronger next feed experiment should
vary stratified slice positions reproducibly by epoch while holding series
selection fixed, preserve physical order/plane coverage, and use deterministic
validation/inference. It is protocol-only; no stochastic-feed production config
or job exists.

The all-series/all-slice224 control already completed at internal0.860340 versus
sampled0.862094 on the identical pool. It did not establish a coverage gain or
test ordered slice interaction. Do not assume feeding more slices without
changing their interpretation must help. Original-source384 rank32 is the
prepared next detail comparison; its full source cache remains unfinished.

Further independent hypotheses are adjacent-slice feature interaction before
study pooling, target-specific spatial evidence from patch tokens, and controlled
multi-plane/sequence interaction. The existing `NeighborSliceContext` has a
zero-initialized residual and geometry-aware ordered messages but is disabled in
these compared runs; a sampled neighbor is not necessarily the immediately
adjacent original DICOM. A true contiguous-neighborhood feed must bind the
source indices and spacing explicitly. Larger detail inputs or ROI crops must
preserve anatomy and the original full-field reference. A montage alone would
compress individual slices and does not establish3D correspondence.

Multi-sequence and cross-plane interaction is supported as a research hypothesis
by [Qiu et al.'s knee-MRI study](https://www.nature.com/articles/s41467-024-51888-4),
on a different cohort. It is not evidence that their scores transfer to this
competition. Prioritize measured comparisons and retain all twelve targets,
source lineage and protected evaluation sets. Backbone size, image resolution,
coverage and context each answer different questions.
