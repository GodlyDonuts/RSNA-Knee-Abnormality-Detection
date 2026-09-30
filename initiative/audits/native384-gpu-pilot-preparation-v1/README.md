# Native384 GPU recovery runtime and verified CPU payload stage

The actual scientific runtime remains clean Git
`c9fcc7eefb09103942be89bf4fa87410eb9ec5ab`, with unchanged accepted labels,
masks, folds and pinned official DINOv3-L weights. The prepared one-H100,
two-CPU, 64-GiB, two-hour allocation has **not been submitted**.

Its paired diagnostic runs use the already accepted 16-training/16-outer caches:
rebuilt256/input224 and original-pixel384/input384, both rank32 at all120
adapter sites, effective batch8, six selected series/16 uniform slices. The
three configs passed the actual TrainConfig validator. The resolution pair
differs only by input side; full versus delta storage differs only by its
storage setting. Actual seeded adapter/head equality, native full/delta/stop/
resume equality, frozen-base checks and strict MRI reload are still required
on the allocated GPU. The helper profiler restores the real checkpoint and
optimizer, warms up once, then measures three eight-study accumulated updates
and inference. Its timings exclude decoding/checkpoint writes, which remain
separately recorded in trainer invocations. Tiny diagnostic metrics cannot
promote a production model.

Local validation passed six exact-inventory/corruption/missing/extra/symlink
cases, config comparisons, helper import/CLI checks and shell syntax. The CLI
check caught an external-helper import path problem before any GPU allocation;
the script now invokes those helpers through runpy from the pinned checkout.
The read-only admission check uses low-thread Git to avoid Newton's observed
thread limit. These checks do not establish actual GPU memory or throughput.

CPU stage **865898 FAILED1:0** after downloading and size/SHA verifying all five
private objects from HF revision
`3f23826ff226992204380d34a2956f3805f8b960`. The failure was a directory-layout
assumption: HF local_dir retained the repository prefix. Original program,
dispatch records, terminal state and read-only inventory are preserved in
`failed-payload-layout-v1/`. The stage was never repeated.

Distinct one-CPU/zero-GPU layout recovery **865906 COMPLETED0:0** in16seconds
checked that terminal failure and every existing byte, then renamed the five
files within the same filesystem. **Zero objects were redownloaded or copied.**
Both archives' **370 regular members each** passed exact membership, size and
SHA checks. Exact prepared script/config/helper bytes also passed. The private
2,027,323,527-byte stage is at
`/home/sa305415/rsna-knee/native384-recovery-c9-pilot-v1`; its exclusive CPU
intents and `cpu-stage-verified.json` prove actual staging separately from the
immutable preparation-time manifest. No credential file was written and no GPU
job was dispatched. Do not repeat either CPU job or the source downloads.

Admission distinguishes CPU staging from GPU dispatch. It accounts for future
deployment copies, all remaining snapshot/recovery/atomic writes, the new cache
stage and diagnostic retention, metadata, and file reserves. Completed privately
verified checkpoint files still retained as unique physical inodes are already
charged in measured Lustre usage; those spent obligations are removed from the
future estimate. No file deletion or hypothetical cleanup is credited. At stage
admission this left **20.668763GiB /6416files** after all obligations. GPU
dispatch was refused because both authorized GPU roles were live. The post-stage
usage was1,017,697,992kB /992,728files; recheck complete projected quota and
actual job states before any later action.

The separate once-only GPU dispatcher is now implemented and locally verified;
see `gpu-dispatch-coordinator-v1/` and
`scripts/newton/dispatch_native384_pilot.py`. Obtain a fresh admission
using phase `staged_payloads`. It must verify this CPU receipt and the exact
prepared bytes, retain a dedicated dispatch intent, admit at most one other
GPU role, and reconcile an uncertain sbatch read-only. Launch only when this
passes. Existing jobs865578/865357 must not be duplicated or interrupted for
this pilot. Original-pixel full-cohort shards and native production remain
outstanding after actual H100 acceptance; do not infer accuracy from staging.

Rank64 E4 /1124updates was also privately fully re-downloaded/size/SHA verified
at `0fd4689e648ece1569cde7a8c030e132e7105bdd`; its separate audit is the rank64
production `epoch-publication/epoch-004/`. Never repeat E1–E4. No rank64 or
whole-series final held-out result exists yet. Submission56711823 still had
null status/score/error at the authenticated18:10:23UTC read.

The source/evidence handoff is on GitHub main **55b85e7** and privately mirrored
at **5a23d61c6de4c7200d74ae732ddbd05d6ef8961d**. All **40 small objects** were
fully re-downloaded/size/SHA verified. This distinct mirror did not repeat
checkpoint publication or the original cache publication. Never repeat it.
