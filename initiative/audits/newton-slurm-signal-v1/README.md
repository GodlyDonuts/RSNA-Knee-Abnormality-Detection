# Actual Newton Slurm signal rehearsals: recovery gate open

Three bounded two-H100/four-CPU jobs tested the production batch USR1 path with the same official DINOv3-L diagnostic configuration. These are failed recovery tests, not competitive runs or Kaggle submissions.

| Stop job | Actual result | What it establishes |
| --- | --- | --- |
| 839401 (`v2`) | Slurm accepted a late signal; trainer finished all 26 updates and the supervisor reported `checkpoint_stop_verified=false` | No proven preemption; the signal came too near normal completion. |
| 839409 (`v3`) | Signal after two updates; batch trap exited 1 because Newton Bash did not define `EPOCHREALTIME` | The added timestamp marker failed before forwarding; no worker stop proof. |
| 839411 (`v4`) | Signal after two updates; corrected batch trap ran and supervisor reported `workers_signaled=2`, but trainer completed 26 updates and `checkpoint_stop_verified=false` | OS-signal forwarding alone did not trigger the trainer's cooperative stop on this actual Newton run. |

Every original small stop script, signal intent/result, scheduler output/error, Slurm accounting output, and available trainer invocation/run JSON is preserved under the corresponding version directory. `source-to-archive.json` binds exact bytes and records the hashes/sizes of the large checkpoints left on authorized private Newton storage. The v4 one-shot watcher had one local setup failure before any signal; its corrected second instance sent the sole v4 signal. No stopped checkpoint was preserved for resume, and no resume job was dispatched. The gate requires an actual terminal exit 75 with a verified stop receipt, followed by a separate same-world-size resume.
