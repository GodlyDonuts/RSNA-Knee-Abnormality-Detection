# Owner-authorized H100 expansion

The owner's October17instruction explicitly supersedes the old two-node ceiling:
use as many accessible Newton H100GPUs as useful; physical four- and eight-GPU
nodes are prohibited. Requesting one GPU on such a node is still excluded.

Fresh `scontrol` configuration, `sacctmgr` association and normalQOS/partition
permissions were read. [Exact timestamped inventory](actual-inventory-and-policy.json)
records physical capacities and exclusions. The account has normalQOS access,
no explicit QOS GPU cap was reported and association MaxJobs is50. Scheduler
allocations/reservations still govern availability. These permissions are not an
assertion that all hardware is idle or that the account owns every device.

The recorded snapshot found one usable idle H100 on evc46, with other usable
H100s allocated; maintenance/reserved devices were not treated as free capacity.
All evc101–104 eight-GPU nodes and evc105 four-GPU node are excluded. Most
assets are node-local on evc33, so scheduling arbitrary nodes without inputs or
runtime does not create useful parallelism. Admit separate small staging and
shards before spreading extraction/training, retaining unique work ownership.

Original spatial scout949626was PREEMPTED0:15 after91seconds, Requeue0/Restarts0;
26completed cases and1,912windows survive. A published-source normalQOS recovery
is prepared to verify/reuse them and execute only the missing six. It has not
been submitted at this source milestone. Its unchanged original numerical path
is separate from future parallel production; no new trained result is claimed.
