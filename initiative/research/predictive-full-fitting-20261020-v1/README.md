# Full predictive B/C/C0 fitting — owner live, waiting for complete inputs

`worker.py.txt` is the single-process, one-H100 worker for one matched arm. **CPU 978976 prepared its exact source, parent, runtime and common initialization on evc33** from source `b361ee40939d875d3adf82d5e04dbf2d7d62b0a0`; see [the actual receipt](actual-cpu-prepared.json). It copied no pixels/models, fitted no model and reserved no GPU. The complete 3,887-study context index still requires separate CPU input admission and fresh arm funding. **No full predictive GPU fit has been dispatched.**

The CPU constructor bound the same shared initialization SHA `dbf69ff87261f7a0657c866ef8e41db8b125a62c6245d80017945c634e60158b` across B/C/C0, with 73,092,652 parameters in B and 73,271,124 in C/C0. Finite fitting owner **59538** is now live, control `307a28b79dddb38a4a61cb997422cd5b30e12dc1`, observed October 20 at 10:52 UTC; [actual start receipt](actual-owner-started.json). It is waiting for the original or missing-only context collection receipts, then will admit inputs once and own the three arms and their independent epoch preservation. Its scientific source remains the actually prepared `b361ee40` across later Git commits. Cache owner **57158** alone owns any cache continuation; this fitting owner must never acquire or regenerate that data.

The owner reads private journals in `artifacts/predictive-full-fitting-20261020-v1/finite-owner/`. After complete inputs and fresh quota checks, it submits B/C/C0 once on ordinary two-H100 evc33, each requesting one H100/two CPUs/eight hours; at most two can run concurrently. It observes progress from original Slurm logs, records the first 200 updates and fixed-final outcomes, and uses one-CPU jobs to independently preserve each sealed epoch. Existing/uncertain intents require read-only reconciliation. It makes no Kaggle mutations and ends by October 22, 09:00 UTC without cancelling healthy jobs.

Fresh admission, dispatch and publication retain the existing shared 96 MiB plus **24 MiB for GPU logs**, totaling **120 MiB/368 files** after usage, along with the 20-GiB/5,000-file floor. Node accounting retains **60 GiB +574 MiB** plus remaining raw/context bounds; selected model caps and paid bytes are counted once. Pessimistic scalar telemetry is 825 bytes ×2,916 updates (2.30 MiB per arm), within each 8-MiB combined stdout/stderr allowance. Excess observed logs stop the owner for review; no process-wide file limit truncates model checkpoints or training.

The explicit numerical decision in `engineering-decision.json` accepts the
zero-mean, full-horizon recipe following actual stability job 978709. Its bounded
24-update result resolves the observed startup spike; peak learning rate and
full-cohort accuracy remain untested.

## Fixed execution

Each independent B/C/C0 run uses all **3,887 unchanged fitting studies**, six
epochs, effective batch eight, **486 updates per epoch / 2,916 total**. Every
epoch has 485 eight-study batches and one **seven-study tail**, averaged by its
actual count in the unchanged trainer. The first three updates are head-only;
all encoder parameters remain trainable thereafter. Encoder/head LR are both
3e-5, density weights B/C .1 and C0 zero. The existing defaults retain weight
decay .02, five-percent warmup, cosine decay ending at .1, gradient clip one and
seed 20261020. `schedule_updates=2916` keeps the full schedule explicit.

The worker calls the bound `Predictive`/`ReconstructionControl` and `train.fit`
modules directly. It reuses the admitted original MRI CoAtNet encoder, replaces
the classification head, uses fixed ImageNet channel normalization, and loads the
exact FP32 24-window/two-mask contexts. Complete source regions and priors remain;
no teacher probabilities, new labels, raw acquisition, protected/gold access or
final fitting-set prediction sweep is introduced. Actual context UIDs must equal
the original sorted fitting UID roster and are joined to unchanged labels/masks.

## CPU-produced private binding required before reservation

Invoke the staged worker only through a later admitted wrapper:

```text
<admitted Python> -B <code root>/worker.py --binding <private binding.json> --binding-sha256 <SHA256>
```

The binding schema is `predictive_full3887_one_arm_v1`. It must contain:

- `arm`, exact `node`, physical GPU count one or two, `source_git`, `code_root`,
  `code_sha256` (including `worker.py` and all imported numerical modules),
  `runtime_python`, optional admitted `python_paths`, and actual `runtime_versions`
  (`torch`, `timm`, `cuda`). `expected_initial_shared_tensor_sha256` must come from
  the admitted exact CPU constructor/parent/seed and match across arms. No node or
  runtime is assumed by this template.
- Small-file `{path, sha256}` references named `runtime_receipt`, `input_admission`,
  `engineering_decision`, `funding`, `license`, `context_terminal`, `context_index`,
  `targets`, and `parent_source`. `parent_weight` additionally has its original
  independently verified SHA and owning-node stat `[inode, bytes, mtime_ns, ctime_ns]`.
- `run_root`, a fresh private directory under `/tmp`; `source_revision`,
  `renderer_sha256`, and `source_context_index_sha256`. A relocated context index
  may have a distinct hash/path/stat mapping only after independent CPU placement
  admission. Its original complete producer receipt remains bound separately.

The input admission must actually have status
`actual_complete3887_predictive_fitting_inputs_accepted`, identical source/code/
node/physical-GPU/common-initialization bindings, SHA fields for `context_index`, `targets`,
`parent_source`, `parent_weight` and `runtime_receipt`, and
`all_context_original_hashes_and_stats_verified: true`. The full producer's original terminal, or its explicitly bound missing-only continuation terminal,
must report accepted 3,887-study exact contexts. The worker
checks original member stats at startup and every load; it does not redundantly
rehash the entire context corpus inside a GPU allocation.

The engineering decision must actually have status
`accepted_full_predictive_fitting_recipe`, exact source/code hashes and complete
`dataclasses.asdict(Recipe(...))`, plus an explicit
`observed_C_C0_gradient_risk_reviewed: true`. This is a required actual decision,
not a value this template creates or evidence of accuracy.

Funding must actually have status `actual_predictive_full_arm_storage_funded`,
this arm and output root, `arm_cap_bytes: 12884901888` (12 GiB),
`standing_free_reserve_bytes` at least 20 GiB,
`other_unpaid_node_liabilities_bytes`, shared free bytes at least 20 GiB and
shared files at least 5,000 **after all liabilities**, with
`all_other_obligations_retained` and `no_deletion_credit` true. No scope is funded
merely by writing this worker. CPU admission must freshly account for data,
runtime, other arms and all existing publication/cache obligations.

## Outputs, bounds and recovery

`epoch_snapshots=True` retains **six original full-state epoch checkpoints plus
`final.pt`**. Each includes full model, AdamW, recipe, exact binding, counters,
numerical history and all RNG state. A conservative tensor-derived bound checks
that all seven plus control overhead fit under 12 GiB; each impending snapshot
and actual output usage is guarded before writing. The full fresh admission check
includes all other unpaid obligations. Runtime checks retain this arm's remaining
cap plus the 20 GiB physical floor, without repeatedly charging other jobs'
subsequently paid bytes as still unpaid. No originals are deleted.

Small sidecars become visible atomically through exclusive hardlink publication;
an existing receipt is never overwritten. Output accounting tolerates removal
only inside the independent publisher's disposable download directories.
The trainer saves each epoch after its update callback, so the worker seals its
sanitized SHA/size/provenance sidecar at the next callback; the final epoch is
sealed after `fit` returns. Sidecars include source, recipe, original parent,
context/target/roster, license, runtime, engineering-decision and funding hashes.
Checkpoint stats are checked before/after SHA; the actual constructor supplies a
canonical model-schema SHA over every state name, shape and dtype for publication
validation. Its shared initialization digest must equal the CPU-admitted value.
No clinical IDs, label arrays or private paths appear in these sidecars. A stopped
job may retain a complete checkpoint before its sidecar; reconcile it read-only,
not by restarting this fresh-only worker. It refuses an existing output root.
A separate immutable recovery wrapper would be required for a genuine resume.

Every update reports numerical losses, gradient norms, LR, actual batch count,
cumulative time/input-load time, per-update throughput, peak GPU/host memory and
current output byte/file counts. The terminal checks all 2,916 updates and **23,322 study exposures**, the
seven checkpoints and six sidecars. Timing includes actual loading and checkpoint
work; it is not a hidden-runtime guarantee. Independent private epoch publication
and full re-download/size/SHA verification remain separate required work. No
publication is performed or claimed by this worker.
