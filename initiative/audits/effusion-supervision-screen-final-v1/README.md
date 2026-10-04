# Fixed-final three-epoch Effusion supervision contrast

Restoring 31 reviewed fitting Effusion cells raised the held-out point estimate
from **0.853370 to 0.894105**, a **+0.040735** change. The paired whole-component
95% interval is **[-0.011129, 0.116394]**: the gain is **not established**.
Only nine observed positive studies/components support this primary endpoint.
The twelve-target macro changed from **0.786090 to 0.787312** (+0.001222),
with conditional 95% **[-0.005310, 0.008490]**. This is an internal weak-label
research result, not a Kaggle score or confirmed MRI ground truth.

Both original continuations **869815/869816** completed **0:0**, three epochs and
843 contiguous finite updates each. The original preempted jobs and their state
remain preserved. Accepted release/masks/folds and protected gold/confirmation
were unchanged. Candidate fitting positives were 47 versus 16 in control.
Both fits used the same rank32 DINOv3-L, native384 inputs, seed20260922,
2,247 training studies and 1,156 disjoint development studies. The completed
ten-epoch baseline is a different budget; do not interpret this screen macro
as a causal regression against its ten-epoch result.

All six immutable epochs were privately published once; all 30 objects were
fully re-downloaded and size/SHA verified. Final checkpoint SHAs:

- Control: `490e23ea1209323a64294b3479affc68ec07918a2ae3b95a7bd62bb8d73a7ffb`.
- Candidate: `5dfae0c57e254ff212fd666b1dcb4a91fa788c2d985593760f7b20793c746269`.

## Original evidence and independent replay

Original run/results/invocation, logs, predictions and sidecars are retained
with the owning-node collection manifest. `slurm-terminal.txt` separately binds
actual scheduler outcomes. The failed read-only observer expected843 records
inside one invocation, but the unchanged trainer retains history per invocation.
The distinct CPU observer869966 correctly joins original updates1–175 and
resumed176–843. Replayed overlap is numerically exact for every loss/gradient/
counter/observed-cell field. No trainer or inference was rerun to repair this
bookkeeping check. See `validation-decision.json` and original observer records.

```bash
PYTHONPATH=. artifacts/qa-env/bin/python -B \
  initiative/audits/effusion-supervision-screen-final-v1/recompute.py
```

The replay verifies original source/full-roster/arm binding, completed original
records, all843 finite updates, final checkpoint/private proof, CSV bytes,
target order and unchanged accepted labels. It recomputes all targets and
4,000 paired leakage-component draws with the predeclared seed20261001.
All primary draws are defined; macro has2,836 defined and1,164 undefined draws.
Undefined draws are retained and reported; intervals condition on defined draws.

## All twelve targets

| Target | Observed + / − | Control AUC | Candidate AUC | Delta | Paired conditional 95% | Undefined draws |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| ACL | 85 / 879 | 0.776062 | 0.783872 | +0.007810 | [+0.003091, +0.012322] | 0 |
| MCL | 2 / 946 | 0.528805 | 0.543076 | +0.014271 | [-0.004386, +0.039170] | 545 |
| Medial Meniscus | 412 / 580 | 0.791632 | 0.791743 | +0.000111 | [-0.001572, +0.001660] | 0 |
| Lateral Meniscus | 149 / 821 | 0.677497 | 0.679545 | +0.002048 | [-0.001095, +0.005308] | 0 |
| Medial OA | 23 / 499 | 0.964581 | 0.964320 | -0.000261 | [-0.003920, +0.003331] | 0 |
| Lateral OA | 9 / 531 | 0.739694 | 0.737184 | -0.002511 | [-0.019197, +0.016417] | 0 |
| PF OA | 19 / 445 | 0.783323 | 0.793732 | +0.010408 | [-0.002778, +0.026192] | 0 |
| Effusion | 9 / 671 | 0.853370 | 0.894105 | +0.040735 | [-0.011129, +0.116394] | 0 |
| Synovitis | 121 / 5 | 0.961157 | 0.966116 | +0.004959 | [-0.002294, +0.018767] | 27 |
| Baker's | 3 / 382 | 0.750000 | 0.744328 | -0.005672 | [-0.029217, +0.009560] | 177 |
| Contusion | 61 / 371 | 0.781671 | 0.785250 | +0.003579 | [-0.001424, +0.008587] | 0 |
| Fracture | 2 / 259 | 0.825290 | 0.764479 | -0.060811 | [-0.097954, -0.032324] | 537 |

All non-Effusion endpoints are secondary/exploratory; twelve simultaneous
target intervals are not multiplicity-adjusted. Do not pick another target
post hoc and describe it as the primary success.

## Next discriminating experiment

Replicate the exact three-epoch control/candidate contrast with one new matched
seed20261004 on the same original fold0, with equal initial states and inputs
within that pair. It tests optimization-seed robustness; it does not add holdout
positives or establish patient-independent generalization. Keep the same
Effusion primary, all12 secondary endpoints, fixed-final selection and paired
component uncertainty. Fold1 would require a separate fitting-only proposal
and full roster admission; the original31-cell fold0 artifact cannot simply be
reused there. Extend to ten epochs only after a coherent replicated effect or
a distinct justified question. No replication GPU has been dispatched yet.

Fresh shared storage has20.0047GiB and4,992 files after retained obligations;
the5,000-file floor is short by eight. New reservation requires a bounded
future-metadata reconciliation, not deletion of original evidence. Original
epoch3 publication exercised node-local HF client storage successfully and
added no shared files; all original home-cache records remain preserved.
