# Predictive final serving — prepared locally, no trained execution yet

This is the deployment path for the new **B/C/C0 full-encoder predictive models**,
not the parent classifier or the untrained throughput probe. Nothing here launches
GPU work, uploads a model/Dataset, pushes a notebook or submits to the competition.
The production models do not yet exist. Local tests establish contracts and output
handling, not model accuracy, T4 precision or hidden-run speed.

## Inputs to CPU compilation

On the private owning node, import `compile.py` and call:

```python
source, receipt = compiler.build(
    original_parent_source_text,
    [
        {
            "publication": original_verified_publication_receipt,
            "provenance_text": exact_original_provenance_json_text,
            "own_source_text": exact_original_own_source_json_text,
            "license_bytes": exact_original_permission_bytes,
        },
        # Optional other actual final arms for the same qualification job.
    ],
    mode="qualification",
)
```

The private originals are the output of frozen publisher
`predictive-full-fitting-20261020-v1/publish_epoch.py.txt`, SHA
`6215f45748fdc4685565a94f31da7fc51f7ad0fa325192d004b362645bbf8d87`.
Its model-only `model.pt` contains exactly `model`, `architecture` and `provenance`.
The compiler requires its actual five-object independent-download publication,
exact small-object hashes, **epoch 6 / 2,916 updates / 3,887 unchanged fitting
studies**, full schema/parameter count, parent/license/recipe/source and common
initialization proofs. Pilots, incomplete epochs, source-derived Stanford weights,
old teacher schemas and altered numerical modules are refused.

The generated source embeds the private licensed parent constructor and our bound
modules. Keep that compiled source on authorized private storage; do not put its
external source body in Git. Only sanitized compiler/actual runtime receipts go to
Git. Compilation neither reads raw MRI nor needs another model download.

## One shared qualification, then one model per competition notebook

The economical plan is **one combined B/C/C0 trained qualification job**. Mount the
independently verified model exports under these unique names, preserving bytes:

- `predictive-B-epoch-006.pt`
- `predictive-C-epoch-006.pt`
- `predictive-C0-epoch-006.pt`

Only mount names for the selected actual exports. The new private model mount still
needs its own justified, funded, verified Dataset or kernel-output attachment;
this compiler does not create or admit that transport. Reuse offline parent Dataset
**12431568 v1** for its original parent construction weight/runtime assets. Use the
same pinned Docker as the throughput probe, two T4 GPUs, four host cores, internet
disabled, and the actual competition data source.

The worker prepares the public three-study MRI contexts once with at most **two
CPU study workers**, then loads each trained arm sequentially. Every complete
learned tensor strictly replaces the initialized state before prediction. The old
parent classifier/logits are never called. Both GPU model copies are released
between arms. Each study receives actual trained FP16-versus-FP32 comparison on its
assigned T4; the first study also runs on the other T4. Predeclared maximum absolute
probability differences are **0.005 across precision**, **0.0001 across devices**.
The measured host ceiling is 22 GiB and each GPU allocation ceiling is 13 GiB.
These checks must pass on actual trained weights; fresh-head probe results cannot
substitute. Qualification produces **no submission.csv**.

After retrieving the actual accepted qualification receipt, compile the selected
single model with `mode="competition", qualification=actual_receipt`. The compiler
binds the exact model, numerical source, worker/contract, observed three-study
coverage and both devices. Changing the serving worker/contract requires a new
qualification; the compiler does not quietly reuse old acceptance.

The competition notebook reads the actual `test.csv`, keeps its original unique
UID order, and uses the same `load_study → assemble → FixedEncoder → predict`
implementation. Each study has at most **24 windows per complementary mask / 48
total encoder windows**, with all original descriptor regions/supports preserved.
There are at most two complete studies in flight. Decimals remain model
probabilities. No report, label, target file, teacher prediction, constant fallback,
UID lookup or remote network input enters inference. Only after all study results
pass does it atomically produce and reparse a **13-column submission.csv**.

The canonical coordinator must separately admit source/version/attachment hashes,
actual public output, fresh account allowance and the eventual code submission.
The public CSV itself must never be uploaded as the competition submission.

## Timing, limits and evidence

The runtime writes sanitized atomic progress at imports, model loading, every
study preparation, every actual precision measurement and each completed model.
The progress file retains only the latest 32 events and aggregate counters;
complete events append once to `events.jsonl`. Logging therefore grows linearly
with the test roster instead of rewriting an ever-growing history per study.
Measurements are retained before acceptance assertions. Failed checks leave
progress and no accepted receipt; there is no automatic retry or fallback.

Qualification reports measured decoding/rendering/FP16/FP32 times and illustrative
1,000/3,887/5,000-study estimates. Three public studies do **not** establish hidden
size, distribution or runtime acceptance. The known serving ceiling remains nine
hours; this template makes no hidden-runtime or accuracy claim. Use the actual
throughput-v2 results and fresh remaining VM budget to choose one combined
qualification cap and useful later public candidates. There is no paid-credit
route, extra engineering run or fixed assumed budget here.

Full raw/context preparation remains a separate bottleneck. This worker preserves
the admitted decoder and renderer; it does not replace raw pixels with a historical
cache, truncate supports or lower resolution to meet time. Source-table ordering
parity was tested separately via the DICOM loader's explicit ordered-file path;
production inference deterministically sorts the available series/SOP paths.
Public trained qualification is still required, and is not a proof that every
hidden data variant is supported.

Tests: `tests/test_predictive_final_serving.py` covers final-publication/source
refusals, complete trained-state replacement, old/partial/nonfinite graph refusal,
actual qualification requirements, exact CSV roster/columns, unchanged encoder AST
and bounded worker count. All test publications and numerical states are synthetic.
