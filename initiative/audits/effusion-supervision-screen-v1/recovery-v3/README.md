# Remove inherited staging CPU bindings

Original869714 failed in one second before either device probe or trainer: Slurm
reported CPU binding outside the assigned step allocation. The one-CPU staging
job exported CPU-binding fields that leaked into the later batch environment.
This is a concrete scheduler launch failure, not model/data accuracy failure.
The earlier869702 driver timeout remains a separate, still-unproven failure.

The distinct v3 wrapper removes only inherited SLURM_CPU_BIND fields from child
environments and explicitly requests srun --cpu-bind=cores. Slurm chooses cores
within each two-CPU step and still isolates its one GPU. CUDA visibility is never
rewritten. A regression test supplies the stale staging mask and verifies removal
without changing GPU/deterministic settings. Original665c0d5 numerical source and
both three-epoch scientific configurations stay unchanged.

The coordinator requires actual FAILED1:0 for869702 and869714 and no original
trainer intents. All previous controls remain intact; it reuses the original
byte-verified code/runtime/cache and private stat proof instead of recopying or
rehashing bulk data. The original32MiB node-local control envelope is split into
30MiB retained originals and2MiB new metadata/journals. Runtime7GiB, checkpoints
3GiB and reserve20GiB remain funded. New shared bytes/files are zero. Submit once,
inspect actual device acceptance and optimizer updates, and never retry uncertainty.
