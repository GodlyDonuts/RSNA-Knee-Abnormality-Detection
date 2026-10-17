# Original S6: actual T4 acceptance and observer recovery

Observed October 17, 2026, 05:50 UTC. The unchanged author source
138285808v2/SHA6e87350d was pushed **once** as private offline qualification
**139165932v1**, `godlydonuts/rsna-s6-original-t4-check-6e87350d5ae0`.
It actually **COMPLETED** on two Tesla T4s in **156.6 seconds**. Strict original
50,018,284-parameter loading, anatomy execution, both original inference paths,
three MRI studies, zero S6 fallback and the finite 13-column CSV passed.
The CSV is 626 bytes/SHA00031c53; it was checked in RAM, not uploaded as an entry.
[Actual acceptance](actual-original-qualification-accepted.json).

This is public-example operability, not a hidden runtime or score. The reported
0.950 remains the author's result. The source still executes CoAtNet even at
weights A=0/B=1; both paths must fit the runtime budget.

## Why the observer stopped after a successful run

The executed script SHA is exactly the pinned author's source SHA. Its embedded
`K1_BUILD_ID` nevertheless differs from the value obtained by running the
author's own `build_id_of` function on that same source. The actual receipt
correctly reports `derived_matches=false`. Our previous observer incorrectly
required true and stopped before CSV acceptance; this was **not a failed GPU
run**. The original source and failed observer remain preserved.

The new observer independently reproduced the stale baked/derived pair and
requires both exact values, the exact executed source SHA (now mandatory), the
live original source returned by Kaggle, and every original model, device,
roster, CSV hash/finite-value and runtime check. It does not accept arbitrary
false build flags. No model or notebook source changed and no qualification
was repeated. [Nine checks and exact pins](source-bound-observer-checked.json),
[actual receipt](actual-original-qualification-receipt-readonly.json),
[canonical observer](phase-observer.py.txt).

## Corrected already-granted compute accounting

Authenticated quota reports totalTimeAllowed=162000s, minimumTimeAllowed=108000s
and isPayToScaleEnabled=true. The actual account UI also displayed 45 weekly
hours and an already linked Colab Pro benefit. The old calculation used the
30-hour minimum and excluded 15 already granted hours. S6 admission now uses the
active granted total while retaining used/reserved time, all original future
fitting obligations, serving reserve and the two-session limit. No subscription,
billing, account or security setting changed.
[Seven accounting checks](granted-quota-budget-checked.json).
The [official Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks)
describes the linked Colab Pro benefit; actual account/API state governs admission.

Old CPU-only observer16380 was stopped after verifying no child was in flight.
A bounded handoff briefly paused only local sampling observer97530, then
**resumed that same PID**; its original GPU training was never interrupted and
no numerical recipe changed. [Actual handoff](auxiliary-lane-handoff-v2.result.json).

## Remaining owned work

Source-bound reproduction owner **20978 / exec44859** adopts the existing accepted
qualification and can push only the one remaining distinct nine-hour offline
notebook. Its latest admission is funded (121772 versus110400 seconds) but waits
for the auxiliary lane while mainE2 and uniformE2 run. A renewable canonical
serving priority is registered. There is no new S6 competition notebook or entry
at this observation. [Admission](actual-offline-admission-wait.json),
[finite owner](finite-reproduction-owner.py.txt).

Distinct sole entry waiter **20994 / exec20982** waits actual offline acceptance,
fresh daily allowance and prior hidden outcomes, then may submit that exact
offline version once. It cannot submit qualification/CPU notebooks or the public
CSV. Old stopped owner intents remain; never restart them. Any uncertainty stops
for canonical read-only reconciliation. [Entry intent](single-entry-owner-intent.json),
[finite entry owner](finite-single-entry-owner.py.txt).
