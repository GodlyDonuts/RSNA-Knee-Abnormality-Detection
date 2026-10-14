# Private source-bound averaged candidates actually built

OneCPU/zeroGPU original step899766.573 built the two predeclared inference-only
candidates from the completed original public arm879172. Original52 source
modules were copied byte-exact into a separate private tree; only the reviewed
weight-average module was added. All original snapshots remain intact.

Epoch10 identity:41939131bytes,SHAc398ddf7c0fca11f62ccef280ca6e458d67cb7c3adf9cc149327b281b0eb1908.
Epoch6–10 effective-weight average:205517883bytes,SHAe2f6cd7409a1edcd86aea31f3e3d4fd4d4ec7d34e36289fdfa0d27c07d1620bd.
Both weights stay privately on evc31; no raw case records, labels, MRI or weights
were exported. The single backbone has rank160 residual factors for the average;
this is exact effective-weight averaging, not factor averaging or a prediction
ensemble. Actual model application/MRI prediction and identity-control replay
still must pass before comparing held-out scores. No accuracy gain claimed.

The separate full localQA first attempt failed at collection because pydicom was
missing from the host environment; the pinned3.0.2 dependency was installed only
into ignored artifacts/qa-deps-20261014-v1. Fresh fullQA87720 is running.
Archive was refreshed for the newly preserved small receipts, without changing
any original dataset, labels or code. Do not repeat candidate build75014.
