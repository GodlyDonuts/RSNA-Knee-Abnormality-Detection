# Reviewed Effusion supervision: parallel fixed screen

The owner asked for actual model training and faster result-driven research.
The accepted numerical source remains `665c0d515410df24f1f37ed5d9f8cb7a0f0fc480`;
actual H100 recovery 869300 already passed complete initial-state and each arm's
full/stop/resume equality. It is reused, not rerun.

The predeclared control and candidate each fit 2,247 studies for three epochs
(843 updates) and evaluate once on the same 1,156 development studies. Both use
native384 MRI, the official DINOv3-L base, rank32 adapters, identical random seed,
views, optimizer and schedule. Only the candidate enables 31 reviewed masked
positive Effusion fitting cells in memory. Original release and outer masks stay
unchanged; report judgments are not independently verified MRI truth.

The new resource wrapper runs two independent one-H100/two-CPU processes in one
Slurm allocation on evc34. Each has 96GiB RAM, safely above the earlier full-run
57–67 million-kB RSS; total 192GiB, four requested CPUs, two GPUs on one node.
The scientific one-GPU/eight-study effective batch is unchanged. Full byte-
verified cache placement 869665 is required before submission. No new GPU
recovery is necessary for a resource-only routing change to unchanged numerical
source and the actual accepted images. Assigned-step CUDA/NVML, original Python
headers, exact source, original assets, and full cache stat inventories must pass.

`dispatch_effusion_screen.py dispatch` is the single coordinator. It stages
owner-only node-local controls via a one-CPU/zero-GPU allocation and submits the
GPU job once. Existing intent means read-only reconciliation, never retry. All
controls/logs/journals live under `/tmp/rsna-effusion-screen-20261004-v1`; new shared
bytes/files are zero. Fresh funding retains shared 20GiB/5,000 files after all
liabilities and funds node-local 32MiB controls, 7GiB runtime, 3GiB immutable
checkpoint/publication work and 20GiB reserve. No original is deleted or credited.

Each completed epoch is privately preserved once using the same coordinator's
`publish --role ROLE --epoch N` command. The bounded publisher pins the original
atomic-lock/private-HF commit/download implementation, adds exact Effusion arm,
source/config/fold/base/license binding, and fully re-downloads/size/SHA verifies
all five objects. Original overlays/case judgments remain private on the cluster.
Both arms' experimental checkpoints are explicitly ineligible for competition
promotion without separate evidence and admission. An uncertain publication is
reconciled read-only; uploads never repeat automatically.

Primary outcome is fixed-final Effusion outer AUC; all twelve targets and macro
AUC are secondary. Report paired whole-leakage-component uncertainty and undefined
draws. Never choose a best epoch or tune against protected gold/confirmation.
This screen can reject the mechanism; a gain requires replication. Launch and
software acceptance alone are not an accuracy result.
