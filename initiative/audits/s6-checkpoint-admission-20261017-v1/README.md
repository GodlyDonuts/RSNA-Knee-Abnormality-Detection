# Actual original S6 checkpoint admission

At **October 17, 2026, 03:13 UTC**, the original public S6 checkpoint is privately
retained on evc33. Original CPU **941506** downloaded it once: **200,185,003 bytes**,
SHA **0c7297d58f061ab4e5d7c55046f51fa3e8934928046f0647a0c7854e98371c12**.
It matches the previously pinned public manifest. No Dataset, notebook, MRI,
GPU or competition entry was created.

The generic final namespace assertion in the first checker failed after the
complete download/SHA and safe CPU load had passed. This was not a corrupt model
or training failure. Separate read-only CPU **941560** inspected the actual S6
state: **362 tensors / 50,018,290 tensor elements / zero nonfinite tensors**.
CPU **941578** reconciled the same retained original file, confirmed its complete
SHA and actual `q_win/cls_w/bias/sp_q/mean/std` graph keys and 12-target head, and
saved the original private tensor header. There was no second download or model
initialization. Original failure and code are retained alongside the successful
[receipt](verified.json), rather than silently edited.

The weights remain at `/tmp/rsna-S6-checkpoint-admission-20261017-v1/E_prior_s42_fall.pt`.
The complete tensor header stays there privately; only its hash and aggregate
counts are published. CC BY-NC 4.0 and nartaa/Heliosli attribution are preserved.
Source Dataset12589750/requested v3 and checkpoint hash are exact. The
[source-only audit](../s6-public-source-20261017-v1/README.md) pins the notebook
and all published code/anatomy contracts.

Fresh admission retained the prior **607,456,016,323-byte** node obligation and
funded a distinct **512 MiB** private checkpoint/metadata envelope, with
630,977,830,912 bytes actually free before acquisition. Shared bytes/files grow
by zero; 20 GiB/5,000 shared-file reserves passed. Actual retained model bytes
are paid once inside that envelope. No deletion credit or GPU/runtime cache
funding is inferred.

**This is safe deserialization and source binding, not module construction,
MRI inference, numerical parity, hidden-scale runtime or a project 0.950 score.**
Before execution, verify remaining exact anatomy/source/dependency assets and
actual hardware/RAM/runtime/output. The source unconditionally runs CoAtNet as
well as S6; normal final rows are S6-only, and bounded unreadable S6 cases fall
back to CoAtNet. Include both pipelines in the benchmark.
