# Frozen C24 evidence attribution

CPU **982349 completed 0:0 in 72 seconds** on evc27, October 20, 2026.
The actual measurement took 68.186 seconds with one CPU, zero GPUs and 2.817 GB
peak process RSS. The checkpoint, original inputs and training recipe were
unchanged. [Actual result](actual-result.json), [terminal](actual-terminal.json)
and [interpretation](actual-interpretation.json) retain the exact evidence.

## Actual finding

At this early 24-update checkpoint, the model uses the target descriptors, but
its readout barely responds to their regional variation on the two fitting
studies measured:

| Intervention | Fitting ordinal 0: mean probability change | Fitting ordinal 1 |
| --- | ---: | ---: |
| Replace regional descriptors with within-case coordinate means | 0.0205 percentage points | 0.00506 |
| Remove the complete evidence branch | 2.516 percentage points | 2.049 |
| Remove the context-prior head | 8.028 percentage points | 7.328 |

Adding the target-dependent score term changes log-Bayes-factor RMS by
0.0551–0.0751. That contribution is present; this is not evidence that the model
ignores MRI. However, the mean-preserving regional intervention has 123–405 times
less readout effect than removing the whole evidence branch. It retains
case-level means and all context-derived predictions, and changes acquisition
variation alongside regional variation. It cannot establish pathology-specific
localization, clinical accuracy or complete context dependence.

Independent FP64 support aggregation matches the original FP32 calculation
within 1.8644e-6. Unobserved support mass remains close to its prior allocation;
the likelihood-scale derivative is nonzero. Neither numerical failure nor a
completely inactive evidence path explains this measured behavior. Marginal
descriptor variances differ greatly across blocks, but these correlated-region
statistics cannot choose a conditional covariance.

**Decision:** retain the original matched B/C/C0 experiment and the separately
declared C100 objective-scale pilot. This result supplies a mechanism to inspect
at fixed final checkpoints; it does not authorize a blind covariance change,
mid-run retuning or promotion based on two fitting cases. There is no new
held-out or Kaggle result.

## Measurement

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

The single actual invocation used source
`2aae5ac934cf91047c6ac9dfbf5f1562472b45bf`. Its historical command was:

```text
python3 -B initiative/research/predictive-evidence-attribution-20261020-v1/run_once.py --source-git 2aae5ac934cf91047c6ac9dfbf5f1562472b45bf
```

Do not repeat it. The runner has one exclusive local journal and no retries. It requests one CPU,
zero GPUs, 12GiB RAM and 30minutes on the owning evc27 node, with stdin source/
payload and stdio output only. It creates no remote files, shared logs, downloads,
checkpoints or account actions. A failed or uncertain invocation retains its
original intent for read-only reconciliation.
