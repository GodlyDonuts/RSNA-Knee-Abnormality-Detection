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

At23:52UTC original raw983422/context983435 remain RUNNING, with emitted
lower bounds3,700/3,887 (95.2%) in each stream. Sole cache/fitting/deployment
processes28954/33107/32378 remain live. No full predictive fit has started.
The matched generative pilot is now terminal and reviewed below. See
[actual live state](actual-live-after-pilot.json).

## Completed generative-routing comparison

Original GPU986011 completed0:0 in50:19. Read-only CPU988297 collected the
exact147,977-byte aggregate acceptance, SHA
`a19883ea4e15574f3563f98a97c792bf742553ef1eafdefadfe345f3ad00bc4e`.
All600 actual updates finished: C100 full200, G100 full200, stop1/resume199.
Identical initial state and complete G100 full/resume state and prediction
bytes matched. Both200-update histories stayed finite through peakLR3e-5.

| Predeclared training window | C100 exam loss | G100 exam loss |
| --- | ---: | ---: |
| First20 | 0.736253 | 0.743408 |
| PeakLR126–150 | 0.291571 | 0.973507 |
| Final20 | 0.288520 | 0.867789 |

These are study-weighted losses on twelve repeatedly seen fitting studies.
Final20 total gradient norms were2.9872 versus61.3908; encoder norms1.3531
versus60.9501, and clipping85% versus100%. G100's final fitting loss was3.01×
C100's and its gradient norm20.55×. Its final density loss was also slightly
higher (0.919301 versus0.919117), with no demonstrated compensating benefit.

**Decision:** reject this G100 recipe for a full run before the deadline.
The routing intervention ran correctly but showed substantially worse fitting
behavior. This is neither a clinical result nor a rejection of generative
learning in general. No four-case engineering predictions selected the recipe.
Keep original B/C/C0 unchanged; no new GPU run or account allocation was made.
All original artifacts and funding obligations remain preserved.

C100 is stable through peak LR, but has no established accuracy gain. Its
full route remains unadmitted and conditional on all primary jobs starting
and an actual non-displacing end-to-end budget. The separate contrast pilot's
metadata graph differs, so its loss is not a causal density-weight control.

[Original acceptance](actual-generative-accepted.json),
[collection receipt](actual-generative-terminal.json),
[window analysis](actual-generative-analysis.json),
[agent decision](generative-agent-decision.json).

Kaggle's authenticated final-selection UI shows0/2 manually selected and
uses the best scored entries for unfilled slots. R6v2 at0.958 remains covered
by this policy, not manually locked. No selection/account mutation occurred.
[Read-only receipt](actual-final-selection-readonly.json).

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
unchanged primary comparison; the completed gradient-routing result is above.
The four engineering predictions did not select this decision.

[Exact terminal](actual-contrast-accepted.json),
[read-only receipt](actual-contrast-terminal.json),
[window analysis](actual-contrast-analysis.json),
[agent decision](contrast-agent-decision.json).


Five local analysis tests passed; these check aggregation and rejection of
incomplete numerical histories, not a model's performance.
