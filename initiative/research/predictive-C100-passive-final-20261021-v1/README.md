# Final C100 passive mechanism observations

Original completed GPU995351 already logged these ten pre-update summaries.
One-CPU1000877 read only those existing aggregates. No extra inference, training,
labels, patient IDs or private predictions were collected. Source8f1e8f2b and
six-epoch final2916 remain unchanged.

The diagnosis-conditioned contribution is active, rather than numerically zero:
its log Bayes factor RMS grows from1.26e-7 at initialization to0.372912 at the
last recorded update. At that update, mean absolute posterior change is0.075384
(7.54 percentage points), and no sampled posterior falls outside[1e-4,0.9999].

On that last **seven fitting studies**, full BCE is0.521508 versus0.559338 for
context-only probabilities (difference−0.037831). The epoch4 seven-study sample
has the opposite sign (+0.014478); epoch5 is−0.005268. These are sparse,
different fitting batches, not a fixed evaluation set or a causal final-model
ablation. They do not establish generalization, diagnostic accuracy, improved
AUC, or a leaderboard gain. No head weight or final epoch was selected from them.

Decision: retain the original fixed final and test the exact candidate. The
mechanism is not wholly inactive, but its independent benefit remains unproven.
Do not infer improvement from the large density-loss constant or fitting loss.
