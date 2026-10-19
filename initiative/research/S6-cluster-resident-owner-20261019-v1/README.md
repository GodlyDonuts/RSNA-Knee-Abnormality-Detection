# Cluster-resident complementary-teacher continuation

October19 actual execution: phase17/969113 COMPLETED0:0 with64new/1,021total studies in1,151.16seconds. Local37032/37034 exited cleanly at the requested boundary. One cluster-resident owner was launched, PID2320176/startticks480928901, control `/home/sa305415/rsna-knee/kaggle/S6-resident-owner-20261019-v1`. Its original phase18 job969332 is RUNNING on evc39;1,032durable rows observed at13:27UTC. No repeat launch or old local producer is authorized.

Actual CPU969208 admitted23workers,47requested/48allocatedCPU,256GiB,128GiB weighted leases,64GiB existing tmpfs cap and32GiB allowance. Original per-study2thread FP32 math, model, renderer, source hashes and population stay fixed. Existing other-user allocation is preserved; unallocated23,265MiB remains above20GiB. Real throughput gain has not yet been established over a full new batch.

`owner.py.txt` runs only Slurm orchestration on login, never model computation. HF token enters once through stdin and remains RAM-only, then flows to original job stdin; no credential file/argv/env. Phase18 depends afterok969113. Up to45distinct64-study phases18–62 run sequentially, stopping on any error/uncertainty without retry. The4MiB/256file shared cap and fresh quota are checked before writes, above20GiB/5000files/32MiB metadata reserve. Later batches no longer depend on a laptop SSH session or VPN.

On actual complete3,887teacher acceptance, the same owner runs the original target constructor on oneCPU/zeroGPU on39, using the already admitted CPU/RAM policy. It retains exact population/model/source/terminal proof and refuses partial rank targets. No studentGPU job is dispatched by this owner. `target_entry_current_plan.py.txt` stages unchanged scientific target code, not a new target recipe.

See actual-owner-started.json, phase17 original receipt in S6-full-CPU-teacher, and original live shared phase intents/results. Read-only reconciliation must precede any recovery; stopped or disconnected observation is never a reason to repeat. Older interrupted render/source429 failures and exact retained-buffer recovery remain in S6-source-quota-recovery-20261019-v1. Prediction completion and engineering recovery are not an accuracy gain.

Actual first resident phase18/969332 has now COMPLETED0:0:64new/1,085total,711.23seconds,23workers. Resident owner automatically advanced to original phase19/969475 RUNNING on39 without local orchestration. Previous12worker batch took1,151seconds, but different study composition prevents a controlled causal throughput claim. See the immutable phase018 receipt in S6-full-CPU-teacher.

Second resident phase19/969475 COMPLETED0:0:64new/1,149total in687.84seconds. Owner automatically continued to phase20/969585 RUNNING on39. This is two observed complete batches at23workers, not a causal same-study hardware comparison or clinical result.
