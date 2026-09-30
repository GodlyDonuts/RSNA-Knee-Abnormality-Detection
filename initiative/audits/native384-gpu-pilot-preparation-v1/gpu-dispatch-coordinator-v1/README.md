# Native384 once-only GPU dispatch coordinator

The canonical command is `scripts/newton/dispatch_native384_pilot.py
--ssh-control SOCKET`. It is read-only by default. Add `--dispatch` only after a
current GPU role frees; it performs an allocated one-CPU/zero-GPU hash acceptance
before making one exclusive GPU dispatch intent and one sbatch call. No native
GPU dispatch or CPU GPU-prerequisite phase has occurred yet.

The CPU proof binds the entire immutable plan, admission program, exact scientific
code and accepted input manifests, all five private payloads, official weight and
config, and accepted environment archive. Every asset has a full SHA proof and
an unchanged inode/size/mtime/ctime signature. Fresh dispatch rechecks source,
inputs, quota and actual GPU roles; changed or missing proof refuses sbatch.
The GPU runtime independently repeats its pinned byte and device checks.

Twenty-four local rejection/transport cases passed, including duplicate intent
or result, actual same pilot job, two live GPU roles, altered/omitted proof,
changed bytes/inode/symlink, failed or unfinished CPU job, source/input/script
changes, fresh admission refusal, and repeat after failed or successful sbatch.
The default CLI created no CPU or GPU intent. These fixtures prove orchestration
behavior; they do not prove native GPU memory, throughput, recovery or accuracy.

The actual 18:51:30 UTC read-only check verified the completed CPU stage and all
nine staged small-file hashes, then returned `waiting_for_gpu_role`: jobs865578
and865357 remained live. It reserved no GPU. Existing stages/downloads/jobs
must never be repeated; uncertainty is reconciled read-only.

The native runtime and 32-study caches remain described in the parent audit.
Actual H100 acceptance and full-cohort native sharding/training are outstanding.
