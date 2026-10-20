# Local predictive-cache deadline analysis

`analyze.py` consumes existing sanitized JSON observations from the original
v3 collector. It performs no remote call, private cache scan, transfer, job or
owner action. It prints JSON and never rewrites input receipts.

```sh
python3 initiative/research/predictive-deadline-observer-20261020-v1/analyze.py \
  --progress \
    initiative/research/predictive-full-cache-alias-recovery-20261020-v3/actual-recovery-started.json \
    initiative/research/parallel-serving-account-20261020-v1/actual-input-progress*.json \
  --scheduler initiative/research/parallel-serving-account-20261020-v1/actual-input-schedule-1351.json \
  --as-of 2026-10-20T13:52:31Z
```

Use the timestamp and scheduler receipt appropriate to each new analysis.
Inputs are bounded to 64 progress receipts, each no larger than256KiB. All must
bind raw979664/context979665, source91224c24 and the exact original intent hashes.
Only completed/total/worker-elapsed/reused aggregate fields are accepted.
Repeated identical sidecars count once; their first observed timestamp is retained
so repeated reads cannot conceal a stalled producer.

The scheduler input is the existing sanitized format `{observed_at, timezone,
jobs:{job_id:{JobId,JobName,JobState,NodeList,StartTime,EndTime,TimeLimit,...}}}`.
`timezone` must explicitly be `UTC`, from `TZ=UTC scontrol` output. The cluster's
default output was EDT: merely treating that naive timestamp as UTC would shift
the wall-limit analysis by four hours. Archived `original_local_time_snapshot`
is never used. Without an actual scheduler receipt, wall-limit slack is unknown;
the helper does not invent a twenty-hour end time.

Raw cumulative throughput excludes the522 retained original cases. Context's
cumulative observed window includes waiting for upstream raw cases and does not
measure independent renderer capacity. The report compares the most recent
interval, cumulative pace, and fastest/slowest pace among the last three observed
intervals plus cumulative pace. Zero progress produces no finite slow-scenario
finish. The combined pipeline uses the later role finish in each scenario.

The spread is conditional arithmetic, **not a confidence interval, guaranteed
upper bound or permission to restart anything**. Unseen study-size tails,
service changes, sealing/index work and future scheduler delays are unmeasured.
Sidecars update only after a completed study; their exact write age is unknown.
An observation older than15minutes is flagged by default, configurable with
`--stale-seconds`. Counts reaching3,887 still require original terminal/index
verification before any consumer admission.

`analysis-1351.json` is a reproducible derivation of the pinned original receipts,
including their small-file hashes. It is not a new cluster observation. Focused
tests cover identity mixing, prefix exclusion, observed slowdown/stalls,
duplicate/stale counters, explicit scheduler timezones, exact actual sample
arithmetic, input bounds and completion versus terminal acceptance.
