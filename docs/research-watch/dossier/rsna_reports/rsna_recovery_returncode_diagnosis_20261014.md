# RSNA patch focus recovery return code diagnosis

Reviewed October 14, 2026, at repository commit `1f05aa1df34bd92212ce92633b99366f98804e3c`. The experiment's pinned scientific source is `bf46b70eea1216673aafb1b5d34e513247b4ef1c`.

## Conclusion

The code explains how completed trainer receipts and a failed recovery wrapper can coexist, but the archived evidence does not establish the exact cause of job 917543's failure. The wrapper rejected the reference/full child process's return value before reading its completed outputs. The JSON field `exit_code: 0` is written inside the trainer before the operating system reports the process's actual exit status.

Keep job 917543 failed and recovery unaccepted. Do not automatically rerun it or convert the completed reference artifacts into full recovery acceptance. Control and candidate full/stop1/resume were not reached, and no clinical improvement follows from these engineering artifacts.

## Observed evidence and its limits

The committed [failure audit](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/initiative/audits/patch-focus-followup-actual-failure-20261014-v1/README.md) reports a working H100, four finite reference updates, completed checkpoint/predictions, completed result and invocation JSON, empty child stderr, and job `FAILED 1:0` after 4:58. That audit contains only a README. It does not archive the child return code, original traceback, raw completion receipts or their file timestamps.

The [dispatch funding receipt](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/1f05aa1df34bd92212ce92633b99366f98804e3c/initiative/audits/patch-focus-evc40-followup-20261014-v1/funding.json) binds the source and wrapper. Independently hashing the fetched pinned wrapper gives `c47a095766efd33878515859591cc7773a8a63eccde1f2470a8857f8673272ec`, exactly matching that receipt. The wrapper, shared executor and trainer have identical contents at the pinned source and reviewed latest commit. The actual launch program is represented by its hash in the dispatch audit; its body and live process state were not inspected.

## What the code proves

### The rejection precedes output validation

The [patch focus wrapper, lines 221–248](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/bf46b70eea1216673aafb1b5d34e513247b4ef1c/scripts/newton/run_patch_focus_recovery.py#L221-L248), expects the reference/full executor call to return integer zero. Any different value raises `Canonical trainer failed; original phase logs retained`. Only after passing that gate does it load the checkpoint, invocation history and resource result. Result-schema, CSV-column, checkpoint-path and recovery-comparison failures cannot produce this particular message in this code path.

### There is no apparent return schema mismatch

The [shared executor, lines 212–270](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/bf46b70eea1216673aafb1b5d34e513247b4ef1c/scripts/newton/run_dense_suffix_recovery.py#L212-L270), returns `subprocess.run(...).returncode`, an integer, rather than a dictionary or CompletedProcess object. It does not consult invocation JSON. The [deterministic entrypoint](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/bf46b70eea1216673aafb1b5d34e513247b4ef1c/scripts/newton/enforce_deterministic_runtime.py#L39-L69) uses runpy without catching or remapping SystemExit. Deadline exhaustion and subprocess timeout raise different exceptions rather than this return-code gate's ValueError.

### Completion receipts precede process exit

The [trainer, lines 1241–1283](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/bf46b70eea1216673aafb1b5d34e513247b4ef1c/rsna/competitive/train.py#L1241-L1283), writes predictions, then completed result JSON, then completed invocation JSON with `exit_code: 0`. It subsequently reads and updates run.json and returns zero. [Run-lock release and cleanup](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/bf46b70eea1216673aafb1b5d34e513247b4ef1c/rsna/competitive/train.py#L581-L599), followed by interpreter teardown, remain after the completion writes. Those writes establish that execution reached that stage, not that the process exited successfully.

The [reference recipe](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/bf46b70eea1216673aafb1b5d34e513247b4ef1c/initiative/experiments/competitive-v1/source-matched-slot-v1/r32-config.json) has `workers: 0`, and the executor forces `WORLD_SIZE=1`. Attributing the failure to a DataLoader worker or distributed barrier is therefore unsupported by this path.

### Ordinary stale output reuse is guarded against

The wrapper [creates the output and role directories without exist_ok](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/bf46b70eea1216673aafb1b5d34e513247b4ef1c/scripts/newton/run_patch_focus_recovery.py#L196-L232), and the executor creates phase logs exclusively. The trainer [rejects a fresh non-resume run containing checkpoint.pt, run.json or outer_predictions.csv](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/bf46b70eea1216673aafb1b5d34e513247b4ef1c/rsna/competitive/train.py#L620-L630). These checks argue against routine reuse of an old full run. They do not replace reconciliation of the actual process, paths and receipt timestamps, which are absent from the failure audit.

## Narrow remaining hypotheses

1. The same child wrote completion receipts, then exited abnormally during the remaining write/cleanup or because of an external termination. This is consistent with the code's ordering but is not proven.
2. The reported receipts or logs were associated with a different path/process, or the actual launch differed from the pinned/default executor path. Existing freshness guards make ordinary stale reuse harder, but the archived aggregate cannot exclude a provenance mismatch.

Neither OOM, a GPU failure nor a particular signal can be inferred from the available evidence.

## Minimum evidence needed from the existing owner

- Original wrapper traceback and phase-marker sequence identifying reference/full and the exact executed paths
- The original child returncode or wait status, if retained. For Python's `subprocess.returncode`, a positive integer is an exit status and a negative integer indicates termination by a POSIX signal; a raw operating-system wait status must first be decoded. Job `FAILED 1:0` describes the wrapper/job outcome and cannot substitute for the nested child's status
- Job plus batch/step accounting and original scheduler stderr around termination, checking for explicit signal, cancellation or resource events. A wrapper step's exit 1 alone still does not identify its child's exit or signal
- Read-only identity, hash and timestamp reconciliation of reference/full's invocation.json, result.json and run.json, phase stdout/stderr, and actual launcher/imported source hashes. No patient rows or checkpoint bytes need exporting

The current gate discards the child return value after comparing it. If no original wait-status or equivalent operating-system evidence survives, the precise cause may remain unknowable without a separately authorized diagnostic. Preserve the original artifacts and uncertainty; do not repeat training merely to obtain a missing return code.

## Scope

This was read-only repository and aggregate-receipt inspection. No cluster access, private case or checkpoint access, training, code changes, submissions, external publication or external-agent contact was performed. No automatic rerun is recommended or authorized by this report.
