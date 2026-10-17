# Preserve the final learned-model entry across allowance windows

Observed October17 07:05UTC. Actual account allowance is four used/one left;
the original E3 entry owner's finite window ends October17 23:30UTC. If that
last slot fills, its existing window would end before the next UTC date. A
follow-on now prevents that deadline from silently dropping the final model.
No server reset or future slot availability is assumed.

Standby **PID28090/exec30473**, local
`artifacts/licensed-parent-epoch-serving-20261016-v1/submit_final_epoch3_next_allowance_once.py`.
It currently verifies original17317 is live and makes **no canonical phase call**.
It never interrupts or takes over a live owner. Only the original terminal
`finite_epoch3_entry_window_expired_without_submission`/zero entries and verified
old process termination permit a later phase. Any original successful entry
exits without a duplicate; missing/uncertain/error outcomes stop for read-only
reconciliation. Process identity, not PID alone, handles PID reuse.
[Intent](next-allowance-epoch3.intent.json),
[actual standby](next-allowance-epoch3.progress.json).

After that explicit release it still waits original final-serving17301/exact
E3 public acceptance. It reuses the **unchanged original canonical entry phase**,
private notebook/version/model, actual1458-update/full-private checkpoint proof,
prior scored outcomes, fresh max5daily allowance, unique entry description and
intent protections. It uploads no CSV, creates no Dataset/GPU job/weight copy,
and makes at most one entry across that follow-on. An uncertain mutation is
never retried. Entry window ends October19 00:30UTC, hidden observation10:30UTC.

Seven actual finite ownership cases pass: live-owner preservation, original
success, original uncertainty, missing public acceptance, finite expiry,
uncertain new entry and actual terminal failure.
[Source](submit_final_epoch3_next_allowance_once.py.txt),
[actual tests](next-allowance-epoch3-checked.json).
This is scheduling readiness, not a submitted or scored final model.
