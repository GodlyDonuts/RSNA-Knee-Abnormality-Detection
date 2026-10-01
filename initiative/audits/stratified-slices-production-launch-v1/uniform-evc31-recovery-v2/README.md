# Exact zero-update uniform recovery on a previously accepted node

Original uniform866715 FAILED2:0 on evc34 in2:31. The allocated-driver probe timed
out (Slurm step exit124), stdout was empty, and there was no header receipt,
initialization receipt, run directory or checkpoint. The underlying cluster-node
cause is unproven. `old-terminal-review.json` and original logs preserve the exact
read-only reconciliation. This is a node-preflight failure, not a model result.

The distinct uniform control was staged once and submitted once as **866719**,
fixed evc31, which previously passed the actual sampling H100 pilot. At first
check866719 was PENDING Priority. The output/control are distinct; do not repeat
its stage/dispatch or reuse its output. The unchanged stratified866716 continues
under the original control. No second stratified job is allowed from this control.

Only the uniform launcher node/name, control/output locations and launch-provenance
manifest differ. Scientific7ac7a97 source, both full configs, input cache, accepted
CPU/GPU receipts and publisher bytes are unchanged. `files.json` binds the exact
new payloads and manual zero-update recovery. The runtime rechecks full source,
assets and actual initialization and writes this distinct manifest into the run.
The single epoch coordinator routes uniform publication to this exact control;
stratified remains at its original control. Private prefixes separate the policies.

A local stage attempt initially stopped before any intent or remote mutation
because it incorrectly addressed the original local stage intent. The coordinator
now selects the distinct audit for both intent and result. An explicit regression
check verifies the old intent is untouched. Read-only recheck confirmed the remote
control was absent before the one successful stage; no uncertain retry occurred.
