# Frozen C24 evidence attribution

Prepared only. No new CPU/GPU job, covariance search, model update or result exists.

This diagnostic reuses the exact C24 checkpoint and cached fitting ordinals0/1
already inspected by CPU981494. The original loader and its actual checkpoint
stat, binding, terminal and numerical-source proofs are pinned. Only its requested
CPU count and Torch thread count change from two to one; its diagnostic callback
is replaced. Both original complementary contexts are used. The remaining ten
training cases and four disjoint engineering cases are not evaluated or used for
selection. The existing sixteen-case target container is read by the original
loader, which selects only rows0/1; no other label row enters this analysis.

The unchanged model predicts μ, δ and context logits a once per context. The
diagnostic reports:

- Per-coordinate target means/variances and negative/positive residual MSE,
  aggregated across observed regions and, for residuals, classes. DCT32,
  gradient16 and texture16 blocks are also summarized.
- Sparse support posterior/prior mass by observed support size, including
  unobserved alternatives; an independent FP64 reconstruction is compared with
  the original FP32 Bayes factor.
- Each support's score is decomposed into `Σδ·x` and
  `−Σδ·μ−½Σδ²`, holding all predictions fixed. Their sum is checked; the report
  includes the mixture log-Bayes-factor change from adding the target term.
  Support scores add, but mixture log-Bayes-factor terms do not.
- The analytic likelihood-scale derivative at β=1. For fixed μ/δ and support
  scores S, `L(β)=logsumexp(logπ+βS)`, so `L′(1)=Eposterior[S]` and
  `L″(1)=Varposterior[S]`. No other variance or β is evaluated on actual data.
- A fixed-prediction intervention replacing each context's observed descriptors
  with their within-case coordinate means. It removes regional variation while
  retaining case-level means. Full, prior-only, evidence-only and intervened
  two-context mean posteriors are compared through aggregate changes and fitting
  BCE, without publishing label vectors or prediction rows.

The exact density loss is
`NLL=(-log p0-q×logBF)/(Nobserved×64)`. Its training coefficient λ does not occur
in inference, which uses `sigmoid(a+logBF)` and averages the two probabilities.
Per-coordinate residual statistics from correlated regions are **not** estimates
of suitable clinical conditional covariance. These two fitting cases cannot
establish accuracy, support independence, or a variance/coefficient choice.

After review and a new immutable commit, root may invoke:

```text
python3 -B initiative/research/predictive-evidence-attribution-20261020-v1/run_once.py --source-git <exact-commit>
```

The runner has one exclusive local journal and no retries. It requests one CPU,
zero GPUs, 12GiB RAM and 30minutes on the owning evc27 node, with stdin source/
payload and stdio output only. It creates no remote files, shared logs, downloads,
checkpoints or account actions. A failed or uncertain invocation retains its
original intent for read-only reconciliation.
