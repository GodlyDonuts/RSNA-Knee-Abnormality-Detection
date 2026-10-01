# Native384: choose the closest completed reference before observing its result

The original native launch named the older sampled224 rank32 result (0.8620941
internal AUC) as its comparison. That older run did not enable strict
deterministic algorithms. A newer, already completed uniform control 866719
provides a closer reference: its actual configuration differs from native
867367 only in model input size, 224 versus 384. The same full official encoder
and adapter provenance, all other settings, accepted labels/masks/folds, 2,247
training and 1,156 development studies, and original environment archive match.
Both actual launch receipts require strict deterministic execution. This
reference is chosen before collecting any native fixed-final predictions.

`replay.py` verifies the original native startup and completed uniform bindings,
all recorded numerical-module SHA256 values against their original C9/7ac Git
objects, exact uniform fixed-final checkpoint/prediction bindings, and both
runtime receipts. Only data.py, train.py and infer.py differ: the preserved diff
adds stratified sampling and maps that policy to uniform during evaluation.
Both compared roles actually use uniform, for which that remapping is identity.
Encoder, adapters, model, loss, optimizer and effective-batch sampler code are
unchanged. The separate original sampling `recompute.py` also reran successfully,
revalidating full fixed-final source/trajectory/roster/masked-AUC evidence and
the predefined 4,000 whole-leakage-component bootstrap draws.

Use uniform **0.8606037925461784 internal macro AUC** as the primary available
reference for the future native result; retain the older baseline as historical
context. No native score, terminal outcome or direct native initial-tensor
equality proof has been collected. This is a best-available comparison with
explicit limitations, not a verified strict resolution-only causal trial.
Native pixels were separately reconstructed. The earlier paired reconstructed256
engineering comparison bounded one uint16-level differences, which does not
prove clinical equivalence or zero training effect. A promising native gain
needs a separately matched repeat before promotion. Do not change the healthy
native job, select a best epoch, or tune on this development pool.

When access returns, first verify the original native terminal, final epoch10 /
2,810 steps, exact source/config/runtime/release/roster and checkpoint/prediction
hashes. Compare all twelve targets and the macro difference against this
reference, pairing whole leakage components and preserving undefined bootstrap
draws. MCL/fracture each have only two development positives and Baker's three;
that uncertainty cannot be removed by extra decimal places or dropping targets.
These are report-derived internal labels, not Kaggle scores. Protected gold and
confirmation remain unused. The accepted masks/folds and both models are unchanged.

Replay from the repository:

```sh
PYTHONPATH=. PYTHONNOUSERSITE=1 artifacts/qa-env/bin/python -B initiative/audits/native384-reference-selection-v1/replay.py
PYTHONPATH=. PYTHONNOUSERSITE=1 artifacts/qa-env/bin/python -B initiative/audits/sampling-fixed-final-v1/recompute.py
```

Newton DNS remains unavailable and Cisco reports Disconnected. This offline
evidence does not refresh the Slurm state or infer that any original job stopped.
No remote job, transfer, publication or competition mutation occurred.
