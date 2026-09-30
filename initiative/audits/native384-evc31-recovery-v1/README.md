# Native384 pilot: driver failure and distinct evc31 recovery

Original pilot **866144 failed 2:0 on evc22** after 1:22. Direct and srun GPU
visibility probes timed out. Stdout was empty; no node-header, initialization,
optimizer update or diagnostic checkpoint existed. This is a device preflight
failure, not evidence about native-resolution training. Never rerun its dispatch.

A distinct evc31-bound script reuses immutable payload hardlinks. Its first CPU
attempt **866157 failed 1:0** because admission still asserted the old control
basename. The original cpu-stage-verified.json was written before full hashes;
it is historical partial evidence, not full-byte acceptance or GPU permission.
The failed intent remains preserved.

The v3 correction changes the basename assertion and uses separately named
staging/prerequisite journals. **CPU866165 completed 0:0** in32 seconds on
evc21 after full hashes of five private payloads, foundation weights/config and
the accepted environment archive. Source, accepted inputs, nine preparation
files, actual jobs and quotas were also checked. It leaves21.5358GiB and6296
files after funded outstanding obligations. No large payload was copied or
downloaded again; no failed receipt was overwritten.

**GPU866172 was submitted once**, oneH100/twoCPUs, pinned to evc31. At22:22UTC
it was PENDING. This is not native GPU acceptance. There are at most two GPU
roles, including whole-series865357. C9 scientific execution is unchanged;
only node, job name and distinct control path differ. Local and remote dispatch
receipts have separate names in this audit.

At22:29UTC the same866172 was RUNNING on evc31. Exact allocated node-header,
device and seeded model-initialization receipts are preserved here; stderr
was empty. Full optimizer/recovery/throughput acceptance is still outstanding.

Inspect actual Slurm/logs and private sidecars before action. Never repeat866172
or prior attempts. When allocated, require real full/delta checkpoint parity,
exact stop/resume Adam/RNG/prediction restoration, strict MRI inference,
unchanged frozen base, and measured native384 throughput/memory. The tiny
diagnostic has no competitive accuracy claim. Full original-source cache
workers and native production remain unlaunched.
