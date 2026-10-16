# Current project state

**Checked October 16, 2026, 08:47 UTC.** This is the canonical research entry point.
Job states and unscored submissions are observations at that time, not promises
about their present state. Exact original receipts and authenticated live state
resolve uncertainty. Earlier observations are in the [history](current-state.history-20261016.md).

## Actual competition results

| Candidate | Submission | Actual public score | Meaning |
|---|---:|---:|---|
| Released author-trained CoAtNet, reproduced unchanged | 57258866 | **0.949** | Attributed external baseline; not an improvement from our own training |
| Prior own DINO model | 57182737 / 57182547 | **0.872** | Best previously verified own-trained score; not the current CoAtNet result |
| Own N3 CoAtNet, epoch 4 | 57256756 | **0.821** | Successful hidden execution, below our prior own best |
| Own N3 CoAtNet, epoch 8 | 57259376 | **Unreported** | Submitted once; latest authenticated observation 08:47 UTC has no status/score |

The own DINO [epoch-5](../initiative/audits/daily-kaggle-comparison-queue-20261014-v1/epoch-005-hidden-scored-reconciled.json)
and [epoch-6](../initiative/audits/daily-kaggle-comparison-queue-20261014-v1/epoch-006-hidden-scored-reconciled.json) receipts establish the .872 tie.
The external result is [canonically reconciled](../initiative/audits/released0949-baseline-20261016-v1/actual-scored-canonical.json).
The own epoch-4 result is [canonically reconciled](../initiative/audits/compact-production-e004-kaggle-20261016-v1/actual-scored-canonical.json).
Zero bytes on an unreported submission do not establish failure. Public three-study
execution and submission acceptance do not establish a hidden score.
The latest recorded account allowance was three submissions used/two available;
query authoritative live allowance before any additional submission.

## What is actually training

| Experiment | Original/current Slurm | Observed state | Scientific change |
|---|---|---|---|
| N3 selected-slot triples, unweighted BCE | 930744 → continuation **932869** | **RUNNING**, evc33 | Original fixed 16-epoch recipe unchanged |
| Source-neighbor challenger | **932957** | **PENDING** | Only donor channels become physical source neighbors where geometry is valid |
| Positive-weight challenger | **933001** | **PENDING** | Only fitting-derived positive BCE term weights change |

Both challengers passed their own actual H100 full/stop/resume checks, including
complete model, BatchNorm, optimizer, scheduler, RNG and prediction equality.
That accepts execution/recovery, not accuracy. They are submitted full runs,
not merely proposals, and no full challenger accuracy is available yet.
See [neighbor receipts](../initiative/audits/compact-source-neighbor-full-fit-20261016-v1/README.md)
and [weighted receipts](../initiative/audits/compact-positive-weight-full-fit-20261016-v1/README.md).

Original N3 stopped gracefully at 2,983/4,496 updates under its explicit boundary
return code 75; continuation preserves the same state and fixed schedule. Do not
count it as a new fit or numerical failure. Original epochs **1–12** have private
full download/size/SHA verification in the publication handoff receipt observed
08:47 UTC. Do not republish them.

## Exact current model and observations

N3 is a **fully fine-tuned 73,136,096-parameter CoAtNet**, not a small frozen-DINO
adapter. Backbone: `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`; finding-specific MIL
attention over 1,024-dimensional image embeddings. Frozen numerical source:
`dcfff319cb7bd7bbae679a20da2b1b9268e8d680`.

- 2,247 fitting studies; 1,156 disjoint weak-label development studies.
- Up to six series, 16 uniformly selected slices per series, 384-pixel inputs.
- Three-channel windows form **after slice selection**; up to 84 windows per study.
  Training samples 12 windows; inference aggregates the full selected bag.
- Fixed 16 epochs / 4,496 updates, effective batch eight; one-study forwards.
- AdamW encoder LR 3e-5/head LR 1e-3, weight decay .02, OneCycle, clip 3.
- BF16 training forward/FP32 loss; no training augmentation in this registered run.
- Fitting labels are dense public soft supervision, not the old sparse hard-label
  release. Accepted release masks/folds and protected evaluation remain unchanged.

Source adjacency, coverage, preprocessing, label semantics and domain adaptation
are hypotheses. No single established bug explains the score gap. The released
0.949 author also describes one-study forwards with accumulation; our microbatch
BatchNorm behavior is not a demonstrated mismatch. See the
[public training contract](research-watch/additional/rsna-public-training-contract-update-2026-10-16.md).

## Data expansion: completed assets versus unexecuted fitting

The additional **345 fitting-only studies / 1,769 series / 54,562 slices** have
completed paired 256/384 cache construction. All seven shards / 42 private objects
were fully downloaded, size-checked and SHA-verified. This is **complete**, not
still being cached. [Original proof](../initiative/audits/additional345-cache-terminal-20261016-v1/all7-private-verified.json).

Source `b009a4a991be508cbe7af2f9b349b1bbef114106` prepares a 2,592-study fitting
population, unchanged 1,156-study outer endpoint and byte-preserving metadata union.
Its local tests passed; the new CPU input gate and expanded training are **not yet
dispatched**. Existing source rows are reused; original 2,247 overlay cells must
remain exact. No clinical label perfection is claimed.

After correspondence exclusions, capacities are **2,592 development fitting,
3,887 train, potentially 4,327 frozen-final**. These are population capacities,
not counts already used in the live run. Sixteen expanded passes would mean
5,184 updates, so that comparison would also increase compute.

## Deployment and immediate decisions

The existing finite curve controller has completed epoch-8 submission and is
observed in **epoch-12 exact public execution, RUNNING**. It later handles epoch
16. Do not restart it, duplicate notebooks/Datasets, or submit public example CSVs.
Epochs 12/16 have no hidden score at this observation.

The 64-study two-T4 raw-MRI benchmark completed in 318.497 seconds with zero
unusable fallbacks and approximately 11.61 GB process RAM. This is measured
engineering throughput, not proof that every hidden study fits the nine-hour cap.

Next decisions: finish the original curve; compare each isolated challenger on
fixed-final held-out evidence; admit the 345-study expansion; then freeze the
final refit recipe. Do not block completed eligible deployment on unrelated
Stanford research. Gold/confirmation remain unused for tuning.

## How to research this repository

Use [research brief](research-brief.md) for findings and limitations,
[continuation](codex-continuation.md) for exact execution bindings, and original
linked audits for proof. Historical markdown is preserved for provenance and
**must not be interpreted as current process state**. Update this page in place
when a result or run phase changes; put chronology in the audits/history.
