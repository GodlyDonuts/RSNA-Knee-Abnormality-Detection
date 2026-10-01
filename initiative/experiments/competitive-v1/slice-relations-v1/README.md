# MRI slice-distance supervision: matched solo DINOv3 experiment

Status: opt-in model/trainer/inference integration and local engineering checks.
The six configs are prepared locally; no relation pilot or production GPU job
has been staged or dispatched. No relation accuracy or Kaggle score exists.
The two live sampling runs retain scientific source7ac7a97 unchanged.

## Mechanism and competing explanations

An examination is a sequence of related measurements of anatomy. A classifier
can fit report labels without learning useful relationships between those
measurements. Test whether predicting physical separation between slices from
their raw image embeddings encourages useful anatomical features in the encoder
adapters. This first version uses global slice embeddings, not registered patch
correspondences or a three-dimensional reconstruction.

The opposite effect is plausible: distance prediction could emphasize scanner,
field-of-view or normal-anatomy cues and distract from a small lesion. An extra
loss might help through generic regularization even when its spatial targets
are wrong. Separate these explanations with physical, shuffled and zero-weight
arms. No teacher, ensemble or per-slice disease label is introduced.

## Prepared numerical contract

All arms restart from the same pinned official DINOv3-L weights, rank32/alpha8
adapters on120 modules, accepted fold0 membership, initialization seed, optimizer
and fixed ten-epoch schedule. They use224-pixel inputs, the same six selected
series and16 uniform slices per series, batch1/accumulation8, BF16 and identical
classification heads. Only the relation weight/target mode differs:

| Arm | Relation weight | Targets |
| --- | ---: | --- |
| zero | 0 | physical distances; same initialized auxiliary head and pair roster |
| physical | 0.05 | actual unsigned within-series distances |
| shuffled | 0.05 | distances permuted within each study/series |

The initial0.05 is a hypothesis, not an optimized setting. Before production,
inspect training-only loss/gradient scales and clipping on the bounded pilot.
If it is numerically inappropriate, preserve that failed acceptance and prepare
one explicitly revised matched protocol before any full-run dispatch. Do not
choose the weight by repeated outer-fold or public-leaderboard probing.

Sample at most32 pairs per selected series using a private CPU generator seeded
from training seed, persisted study UID and epoch. Pair rosters survive batching,
DDP rank partition, accumulation and resume. The shuffled arm shares the same
roster and preserves each selected series' target distribution. Unknown geometry,
padded slices/series and distributed tail duplicates contribute no targets.
Actual geometry availability passed separately for all2247 training studies and
11639 selected series; that survey is not a model evaluation.

Targets are log1p(unsigned millimeters)/log1p(100). A symmetric MLP consumes only
the absolute difference and mean of two raw encoder embeddings, before position
or protocol embeddings. Physical metadata supplies targets only. The128-unit
head adds262401 parameters for1024-dimensional DINOv3-L embeddings; the original
pretrained tensors stay frozen. No extra encoder pass is required.

Average Smooth-L1 relation loss over pairs within each represented study, then
over represented studies across the entire effective batch and all ranks. Add
its weighted contribution to the existing masked per-target classification loss.
A rank with no eligible geometry still connects every auxiliary parameter to
the graph with zero contribution. Individual slices receive no disease labels;
the twelve disease predictions remain examination-level outputs.

## Acceptance and resource gates

Local checks include unchanged shared initialization/RNG and zero-weight
classification gradients, metadata/target separation, UID replay, accumulated
versus full-batch gradients, actual two-process CPU DDP with a zero-geometry rank,
frozen-base delta/AdamW recovery, trainer mid-epoch/external-stop recovery for
physical/shuffled/zero arms, and MRI-only synthetic-cache inference with the
auxiliary forward explicitly forbidden. These are engineering evidence.

Next require an exact committed source pack and official-weight real-MRI pilot
using the prepared16-study/two-epoch configs. Bind accepted training UIDs, cache,
release, folds, environment and weights. Check equal initial classification
states/logits, real finite optimizer progress, adapter gradients, frozen original
tensors, loss/gradient scales, decoded interrupted-resume equality and twelve
MRI-only outputs. The tiny pilot is not competitive performance and cannot
initialize a full production arm. Full224-pixel BF16 GPU acceptance remains
outstanding; CPU tests cannot establish H100 throughput or T4 deployment.

Do not reserve a third GPU while sampling866716 and866719 are running. Use at
most two concurrent GPU nodes and minimum CPUs; serialize the third relation
arm. Fresh byte/file quota must fund all future snapshots, checkpoint/optimizer
state and publication verification while preserving20GiB/5000-file reserves
and outstanding native384/sampling obligations. No new source image cache is
required. Existing sampling publishers are pinned to their old protocol and
must not be repurposed for relation checkpoints; prepare a separately bound
publisher before a relation production launch.

## Promotion or rejection

Use uniform MRI-only evaluation and fixed epoch10 on the same1156 outer-fold
studies. Compare all twelve targets, macro AUC and leakage-component bootstrap
uncertainty; report undefined rare-target draws. No best-outer-epoch selection,
protected confirmation/gold use or changes to accepted masks/folds.

A physical arm that fails to improve against zero does not support this
mechanism. Similar physical/shuffled gains support a generic auxiliary-loss
explanation instead of spatial information. Inspect variance of raw embeddings
on the admitted training probe and relation error alongside disease performance;
loss reduction alone is insufficient and feature collapse rejects the candidate.
Repeat a promising matched gain on another fold or seed before a broad claim.
Kaggle submission requires independent exact offline runtime acceptance and a
genuine production candidate; diagnostic outputs are never submitted.

The existing uniform sampling arm is useful context, but the fresh zero-weight
arm is the primary control: it uses this exact integration source and auxiliary
head. Three full arms need roughly three current-run training budgets; the
pilot must measure actual extra time/memory before scheduling them.
