# Count preparation memory before accepting deployment

The original private FP32 prefetch32 result reported 8,157,212,672 bytes by
adding parent and two resident-model peaks. It omitted both persistent CPU MRI
preparation workers and the multiprocessing resource tracker. Preserve its
exact-prediction and timing evidence, but do not call that RAM figure a complete
process-tree bound. The unchanged running 1300-study notebook137329288/v1 has
the same limitation; no replacement push or repeated inference was performed.

The new reporter reads Linux `VmHWM` for the parent, both resident workers,
both preparation workers and the resource tracker **before worker shutdown**.
It discovers descendants across all threads, binds the original live worker
PIDs, and rejects missing or unclassified processes. Summing lifetime peaks is
a conservative bound on these processes' simultaneous RSS. Current RSS after
freeing a buffer is insufficient. No additional library, MRI processing,
model arithmetic, output ordering or checkpoint change is required.

Actual one-CPU/zero-GPU Linux job877784 verified all six live processes after
four 64-MiB worker buffers had been freed. The reported upper bound was
343,834,624 bytes, including 155,615,232 bytes for preparation workers and
14,176,256 for the tracker. The exact accepted helper source SHA is in
`v3/actual-Linux-CPU.json`. This is an engineering memory-accounting test,
not real-MRI/T4 memory acceptance or clinical evidence.

Original job877775 failed before helper execution because the runtime was
node-local on evc31, not evc32. Distinct877776 reached the helper on evc31 but
failed on an unavailable psutil import. Both failures are retained; v3 uses
only the standard library and original owning-node runtime. No existing job,
notebook, checkpoint, label, mask or fold was changed or restarted.

Twelve focused tests cover kernel high-water marks, complete descendant
enumeration, omitted workers, unclassified processes and worker-role overlap.
Existing spawned MRI preparation tests still verify original selected tables,
cache bytes and recovery receipts, and now inspect both live preparation PIDs.

Before a future deployment or memory acceptance, package the new helper with
the probe and collect actual MRI/T4 measurements using this reporter. The
current fullscale run remains useful for exact output and elapsed time; do not
repeat it merely to reproduce the old FP32 reference. This reporter bounds
workload process RSS, not total cgroup/page-cache usage or notebook-bootstrap
RAM, and does not establish that the hidden test will finish within nine hours.
