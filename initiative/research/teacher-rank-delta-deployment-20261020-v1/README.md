# Ranking model: one public run, then the same code submission

This distinct route tests the completed ranking checkpoint while the predictive
B/C/C0 inputs continue preparing. It does not restart training or the older,
unlaunched ranking deployment. **Entry 57450065 is COMPLETE, public score 0.948,
no error.** Source-zero, pointwise and ranking have the same displayed 0.948;
that does not imply identical predictions or an improvement over our 0.949
adaptation or external R6’s 0.958. See the [actual outcome](actual-hidden-outcome.json).

The ranking and pointwise models share the serving compiler, graph and MRI
preprocessing. Pointwise already passed the original 32-study test on both T4s.
Repeating that entire code check for different learned weights is unnecessary.
This route inherits that code evidence and separately checks the actual ranking
weights on **all three official public studies, both 94-window views and both
T4s**. It requires identical CPU/GPU inputs, complete learned-state equality,
finite predictions and at most `1e-6` FP16 serving-versus-training graph error.
It does not claim that ranking passed 32 studies or that these checks measure
accuracy.

The self-test runs only for the exact public roster. The same immutable notebook
then runs its original competition inference and produces `submission.csv`.
After successful source, receipt and CSV verification, the canonical coordinator
submits that exact code version once. The hidden roster runs the original
inference without repeating the public self-test. Failures cannot produce a
fallback CSV for submission.

The accepted allocation is a 180-second public watchdog plus 80 seconds for VM
startup, preserving **3,360 GPU-VM seconds** for predictive deployment. The
prelaunch account observation had 3,672.737 seconds, leaving 52.737 seconds beyond
both allocations. The actual post-entry account retains 3,587.508 seconds,
including the 3,360-second predictive allocation. Startup is a funded allowance, not a guaranteed upper bound;
the actual charge must be inspected afterward. No paid credits or uncertain
retries are allowed.

The immutable ranking model is
`d478a502e6fd33be2ad9d781acd91d3d2b6b252629262031fba8282a91cf5bca`,
epoch 3 / 1,458 updates from original job 975297, privately preserved by
CPU979757 and compiled by CPU979767. The Dataset is created and fully
download/size/SHA verified once under the existing canonical account claim.
Current storage obligations and all healthy training remain intact.

See [the explicit engineering decision](engineering-decision.json) for the
original evidence, narrower test scope, budget and residual risks.

Actual read-only CPU982654 compiled the exact private original source with the new wrapper in RAM: source `b0d5303368fd0140167112c9946fcd35895bb82ba8ead98038eb40fb3594d71b`, 135,665 bytes, zero remote writes. The 20 serving and 21 deployment tests pass; the latter exercise canonical state transitions with real temporary SQLite and fake services. These are engineering checks, not a T4 run or a score.

Actual October 20 19:23 UTC: private Dataset **12739137 v1** was created once and all seven objects verified. The original owner stopped after a read-only RPC error; the same original Dataset was reconciled without another create. Continuation **20168** then ran notebook **139878511 v1** once: all new-weight checks passed on both T4s, the output had zero fallback/decoding failures, and public execution took **73.83 seconds**. Code submission **57450065** is **COMPLETE, scored 0.948, no error**. Continuation 20168 recorded the actual terminal result and its process has ended; do not restart it. The measured free-account difference was **85.229 VM seconds**, leaving **3,587.508 seconds**, including the intact **3,360-second predictive reserve**. All five current-day entries are now used. Do not repeat this Dataset, run or entry. See [the reconciled summary](actual-deployment-summary.json).
