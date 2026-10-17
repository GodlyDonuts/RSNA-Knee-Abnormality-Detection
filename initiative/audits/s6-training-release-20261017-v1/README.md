# Released S6 training recipe and fitting-label comparison

Actual October17 09:57–10:09UTC. This is a new source/target finding, **not a new
competition score**. Best reproduced S6 remains0.950; the owner targets0.97.

## What became available

`nartaa/rsna-knee-s6fall-weights` Dataset12589750 advanced fromv3 tov5.
Training code was added in v4; v5 corrects the local pool explanation and code
licence. Original CPU945294 acquired only missing small source/manifest/label
files,3,669,231payload bytes, under a16MiB node-local cap. Five already-complete
files were reused. The original nested-filename GET failed before completion;
its failure is preserved. The distinct finish used the official SDK's versioned
DownloadDataset method and fetched only missing files. No model was downloaded
or rehashed. [Exact original versioned file proofs](acquired.json).

The scored S6 trainer `train_s6f0_all.py` SHAae18d2e1 uses ConvNeXt-small from
pinned ImageNet-22k initialization, per-finding spatial/window attention,
320-pixel preparation/288crop,3adjacent channels,3.5mm physical spacing,
64uniform training windows, batch2,16epochs, encoder/headLR1e-4/1e-3,
AdamW0.05,5%warmup/cosine, gradient clip5 and gradient checkpointing.
It trains4,263studies, fixed-final epoch16, noEMA/SWA or Gold-based epoch
selection. These are original released recipe/source declarations inspected
against the acquired code/config; we have not reproduced its training.

The released S6 has **no OAI training**. The separate CoAtNet0.949 source used
2,399OAI knees with onlyPFOA/LateralOA/Synovitis supervision and Gold-ranked
snapshot averaging. Do not transfer its provenance or selection claim to S6.
The new code declares Apache-2.0; S6 weights retain CC BY-NC4.0 and attribution.

**Corrected pool fact:**86competition studies were removed from the author's
local disk for storage. They were not removed by the competition host and were
not a quality filter. The frozen4321pool contains4263fit+58Gold. Our3887fit
includes73studies absent from that local pool. This does not admit new clinical
labels or alter our accepted cohorts.

Primary source: [author's training-code dataset](https://www.kaggle.com/datasets/nartaa/rsna-knee-s6fall-weights/versions/5),
[author's correction and exact reconstruction discussion](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/750180).
Raw source/targets stay on private Newton storage; only hashes and aggregates are
archived here.

## Actual label comparison, CPU945356

[Original aggregate receipt](fitting-label-comparison.json) binds every source,
UID roster and target hash. Current fitting targets are Stevenv6/eb4c2b29;
S6 uses `labels_v1_blendjev`, declared as a cellwise mean of the author's
`labels_v1_blend` and JEV HYBRID report labels, keeping the first where JEV is
missing. The released rebuild script and inputs are pinned. Neither set is
clinical ground truth or independent of upstream recipe development.

All3887×12current cells and released fitting cells are observed. **Zero cells
match within1e-6.** Correlations below describe source-label ordering only.

| Finding | Mean absolute target difference | Rank correlation |
|---|---:|---:|
| ACL | .06220 | .94588 |
| MCL | .06843 | .84396 |
| Medial meniscus | .02474 | .97985 |
| Lateral meniscus | .02570 | .93092 |
| Medial OA | .06295 | .96667 |
| Lateral OA | .06746 | .95691 |
| PF OA | .07354 | .97414 |
| Effusion | .08798 | .98761 |
| Synovitis | .04471 | .97755 |
| Baker's | .07361 | .93520 |
| Contusion | .04053 | .95379 |
| Fracture | .09278 | .51446 |

Fracture is a clear supervision difference worth isolating; this does not prove
that changing it will improve the leaderboard. Skip all520nonfitting rows before
numeric conversion. The58public Gold rows have structurally blank targets;
no protected clinical numeric values were read. Existing accepted labels,
masks, folds and healthy recipes were untouched.

## Concrete next comparison

CPU945389 materialized one distinct613,297-byte fitting-only target overlay,
SHA5e34bbf4, privately within the existing16MiB release envelope.
[Actual full binding and round-trip equality](fitting-overlay-accepted.json).
It preserves exact UID order, all masks and every two-view parent logit; only
the source-supervision array changes. The original targets are unchanged.

A label-only arm must match the original whole94 starting checkpoint,3887fit,
94windows,3epochs/1458updates, optimizer/sampling/RNG and25%source/75%parent
loss, substituting only this target array. Do not call it full S6 reproduction:
architecture, cohort and observations differ. Actual new input/recovery,
checkpoint/VM funding and GPU admission remain prerequisites. No Dataset,
trainer, checkpoint or competition entry was created by this label preparation.

The weaker anchoring versus stronger supervised optimization is a **separate**
mechanism. Do not change both labels and loss weight in a purported label-only
comparison, or modify the currently running original fits. A source-exposed
internal cohort is not whole-model independent OOF; actual held-out results
must decide promotion. This is a falsifiable next contrast, not a0.97 claim.
