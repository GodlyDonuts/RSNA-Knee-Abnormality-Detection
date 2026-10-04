# Second matched seed: reviewed fitting Effusion supervision

Original control870025/candidate870026 completed3 epochs/843 contiguous finite
updates each, seed20261004. Actual small original result/prediction/log/sidecar
collection required complete private final-epoch proofs first. No training or
inference rerun, checkpoint rehash, mask/fold or protected-data change.

Effusion AUC0.928051→0.941795 (+0.013744), paired whole-component95%
[-0.012795,0.045965],9 observed positives/671negatives. The improvement is
unestablished. Overall12-target macro0.809696→0.801626 (-0.008070), conditional
95%[-0.014051,-0.001754],2836 defined/1164 undefined draws. The candidate harmed
aggregate performance in this seed despite improving the primary point estimate.

The first seed showed primary+.040735 and near-null aggregate+.001222. These
same-fold seed repeats do not establish an overall win or independent patient
generalization; retain original-label control as the default. The owner's new
Kaggle priority admits all four fixed-final checkpoints as a paired diagnostic
comparison of trained production-size models, not promotion of the overlay.

Run PYTHONPATH=. artifacts/qa-env/bin/python -B
initiative/audits/effusion-supervision-seed2-final-v1/recompute.py for the exact
original source,roster,trajectory,12-target masked-AUC and4000-draw replay.
comparison.json retains every target and undefined draw count. No Kaggle score
is inferred from internal results.
