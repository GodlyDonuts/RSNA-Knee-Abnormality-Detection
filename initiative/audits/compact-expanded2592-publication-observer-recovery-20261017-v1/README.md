# Expanded N3 publication observer recovery — October 17, 2026

The old replacement observer **93375** stopped when its child could not import
`scripts.hf` from a direct script invocation. The traceback occurs during module
import, before `main()`, credential access, intent creation or a publication RPC.
Epoch-1 local publication intent/result were absent when this was reconciled.
This is an observer import failure, **not a failed model fit or lost checkpoint**.
Original Slurm **938284** continues unchanged with its epoch-1 snapshot intact.

A distinct finite observer **95408 / exec49302** pins the same coordinator bytes,
sets the repository import path and disables user-site imports in the child.
Actual import resolution was checked before starting it. It keeps the original
per-epoch no-duplicate checks and performs no GPU launch/restart. The corrected
observer passed the original CPU/private-path prechecks, created the first actual
publication intent once and submitted CPU-only **941066**, last PENDING Priority.
An accepted intent/queued CPU is not a verified private checkpoint publication.

Read [original-failure.json](original-failure.json), the
[exact replacement observer](watch_original_epochs_with_repo_imports_once.py.txt)
and [actual-recovery.json](actual-recovery.json). Existing intent now requires
read-only reconciliation if publication is uncertain; never repeat this dispatch.
Use the original publisher and watcher to collect the later full download/hash
proof. All numerical source, data, optimizer and original training remain unchanged.
