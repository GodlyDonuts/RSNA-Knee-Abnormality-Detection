# Bounded allocated-driver recovery

Original869702 failed before either trainer command or optimizer update; both
assigned steps exited124 at the25-second read-only CUDA/NVML guard. No root cause
is proved: a short bound can conflate slow cold initialization with a hung driver.
A distinct wrapper retains every original and extends the finite driver bound to
180seconds, printing scheduler context, NVML return and before/after cuInit stages.
It never changes device visibility or bypasses CUDA checks. Source665c0d5,
configurations/images/optimizer/schedule remain byte identical.

The distinct v2 single coordinator requires actual original FAILED1:0, both124
step results and absence of trainer intents before staging or submitting. It
reuses the original byte-verified6.3GB runtime and exact source on the same node;
there is no repeated runtime/cache copy. Owner-only node-local controls split the
original32MiB envelope into16MiB retained originals and16MiB v2. Both CPU-only
publication and training retain3GiB checkpoint and20GiB reserve obligations.
No new shared bytes/files. Existing intents remain immutable; uncertainty is
reconciled read-only. If the extended guard fails, inspect its actual stage and
node evidence before moving compute; do not claim model/data failure or progress.
