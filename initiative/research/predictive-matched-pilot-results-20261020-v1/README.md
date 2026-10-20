# Original matched-pilot result collection

`collect.py contrast` targets only original GPU984784. `collect.py generative`
targets only original GPU986011. A live job returns its actual scheduler state
without an intent or CPU allocation. A successfully completed job is read once
using one CPU and zero GPUs on evc27. No job is submitted, restarted, cancelled,
or promoted by this collector.

The complete original aggregate acceptance bytes are retained without
reserialization. Their source, binding, scheduler completion and run-start
identity must match the original submission. Weights, case IDs, clinical
labels, individual predictions and MRI arrays are not collected. Any existing
collection intent is reconciled rather than rerun.

The report computes the predeclared first20, peak126–150 and final20 update
windows for both roles. It presents update means and study-weighted means
separately, because the actual pilot batches alternate eight and four studies.
It preserves clipping rates and gradient norms. These are training diagnostics
on twelve fitting studies, not clinical accuracy or automatic promotion.

At23:00UTC original raw983422/context983435 remain RUNNING, with emitted
lower bounds3,400/3,887 (87.5%) in each stream. G100986011 is now RUNNING;
its original startup and32 finite C100 updates were read on the owning node.
See [actual live state](actual-live.json).

## Completed contrast comparison

Original GPU984784 completed0:0. One read-only CPU986874 collected the exact
142,002-byte aggregate terminal, SHA
`1e15ae52a8d43c5e8aeebfbdcb3a47324181815c366359f2ffa313280e40afa9`.
No weights, individual predictions, labels or images were collected.

| Predeclared training window | Control exam loss | Normalized exam loss |
| --- | ---: | ---: |
| First20 | 0.661904 | 0.690622 |
| PeakLR126–150 | 0.295191 | 0.292573 |
| Final20 | 0.283679 | 0.284710 |

These are study-weighted losses on twelve repeatedly seen fitting studies.
Both roles completed200 finite updates through maximumLR3e-5, with actual
encoder/head parameter changes. Final clipping was65% control versus85%
normalized. Both density losses remained around0.9196. The normalized role
completed stop1/resume199; a new GPU full-versus-resume comparison was not run.

**Decision:** defer the full contrast cache and fit. The pilot shows no
consistent optimization advantage sufficient to justify another full input
transformation before the deadline. It does not establish that normalization
cannot improve held-out accuracy. Preserve the originals and prioritize the
unchanged primary comparison plus the already running gradient-routing pilot.
The four engineering predictions did not select this decision.

[Exact terminal](actual-contrast-accepted.json),
[read-only receipt](actual-contrast-terminal.json),
[window analysis](actual-contrast-analysis.json),
[agent decision](contrast-agent-decision.json).


Five local analysis tests passed; these check aggregation and rejection of
incomplete numerical histories, not a model's performance.
