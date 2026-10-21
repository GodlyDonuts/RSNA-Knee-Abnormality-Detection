# C100 fixed-final deployment

Prepared October 21, 2026, while original full fit **995351** continues on evc27.
This is executable, locally tested deployment code. **No C100 Dataset, notebook,
qualification, competition entry, new job or model transfer has been dispatched.**

C100 uses the same C architecture, original initialization, fitting order,
accepted labels/masks and six-epoch schedule. Its density-loss coefficient is
100 rather than 0.1. Source is `8f1e8f2bb9f2ded8a2e4a582718188b68639db8b`.
The actual training result is still unobserved. This path consumes the actual
final model; it cannot substitute pilot 981620 or an intermediate epoch.

## What this implementation does

- Builds a separate single-model deployment on **evc27**, directly from
  `/tmp/rsna-predictive-full-C100-20261020-v1`. It does not transfer weights back
  to evc33 or change the active B/C/C0 deployment owner.
- Reuses the frozen canonical deployment at
  `fb9e3adf2d4f2f873adf10733bb310e7afa73be8` and the already tested
  [C100 serving compiler](../predictive-density100-serving-20261020-v1/README.md).
  The inference worker, decoder, graph, precision checks and CSV writer stay
  byte identical. Only branch identity, compiler adaptation and account-budget
  protection are added.
- Uses the existing canonical Newton ledger, shared account claim, explicit
  single-operation intents, fresh allowance and exact private Dataset verification.
  Dataset/notebook slugs and ledger family distinguish C100 from C and G100.
- Preserves **3,360 GPU-VM seconds** for B/C/C0. The additional reservation is
  released only when the canonical ledger contains three accepted code entries
  from the same primary bundle, including the exact original B/C final hashes.
  A caller-provided zero reservation, missing receipt, different bundle or
  uncertain submission cannot retire it. Other reservations remain intact.

## Execution after actual completion

Use `run_once.py --help` for the original explicit phases: `stage`, `prepare`,
`dataset`, `dataset-reconcile`, `compile`, `push`, `status`, `submit`, `outcome`.
The modes are `qualification` and `C`; the latter remains internally architecture
C but is externally distinguished as **C100**. The separate [finite owner86015](../predictive-C100-deployment-owner-20261021-v1/README.md) now waits for the actual original final and drives these explicit phases. There is no automatic mutation retry.

Create a private, token-free configuration only after reviewing original job
995351 and its completed publication journal. Alongside the original deployment
fields, it needs:

- `C100_contract_options`: actual full source, actual derived publisher SHA,
  exact accepted pilot receipt text/SHA and original pilot scheduler record.
- `C100_final_evidence`: all six ordered private publication receipts, the
  original full terminal record and actual successful terminal scheduler record.
- `owning_node: evc27`; one `original_runs.C` entry pointing at the original
  C100 run and original license; a unique root beginning
  `/tmp/rsna-predictive-final-deployment-20261020-v1-C100-`.
- Fresh explicit storage funding for the existing deployment policy's 2 GiB
  node-local and 1 MiB/eight shared-file scope, retaining all other liabilities.
  Nothing in this preparation allocates new storage or retires prior obligations.
- Fresh runtime/account decisions before GPU use or submission. C100 requires its
  **own actual trained qualification**; B/C/C0 or Colab measurements do not replace it.

`stage` refuses incomplete production evidence. `prepare` rereads the actual
final five-object publication and scheduler on its owning node. The serving
contract checks every epoch, source/recipe/initialization binding and full
2916-update outcome before compiling. The competition phase binds the exact
qualified model, Dataset, source and precision result. An uncertain mutation
must be reconciled read-only.

## Validation

**76 scoped tests passed**: 19 new deployment tests plus 57 existing C100
contract/compiler tests. These execute the generated staged compiler, compile
qualification and competition code, reject an unrelated qualification or
incomplete/pilot final, check the unchanged canonical account implementation,
and exercise budget retirement against an in-memory canonical ledger. Ruff and
the explicit CLI help passed. These are software checks, not a GPU qualification
or an accuracy claim.

## Actual training observation

At **10:29 UTC**, original C100 had **182 contiguous finite updates**, no stderr,
and full encoder updates. Passive observations at step146 showed mean absolute
posterior change **0.05812825** from the evidence term and training-batch BCE
**0.585147** versus **0.603624** from context alone. This confirms an active
evidence term on eight training studies; it does not establish clinical accuracy,
generalization, improvement over C/C0, or a Kaggle score. Different observed
steps contain different batches and are not a matched learning-curve comparison.
No extra forward/backward passes or model changes produced these diagnostics.
[Original observation](../predictive-density100-full-20261020-v1/actual-peak-LR-progress-20261021.json).
