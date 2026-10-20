# Pointwise deployment and completed ranking milestone v8 — prepared only

No manifest is finalized and no publication has been attempted. Root must first
provide the final reviewed Git pin containing the actual pointwise qualification
push and Dataset verification. Never publish while the pointwise account owner
is active, even between its short API phases.

The predecessor is exact source `17beffadba6e7574bd6da69dbac1de9992e2bd05`,
fully verified once at private HF revision
`91ed4964850174a5c8f677b2a4f9af6e4e979a57`. Do not repeat v7.

The explicit committed delta includes our pointwise deployment/continuation
controls and tests, its private Dataset identity and byte-verification receipts,
actual qualification push, read-only account/input/resource observations,
completed ranking epoch-3 preservation and CPU compilation, current guides,
and the preceding mirror's verification chain. Only actual receipts present at
the final pin are included; a push is not qualification acceptance or a score.
Separate ranking deployment and C0 relocation preparation are outside this scope.
Raw MRI, reports, clinical IDs, individual predictions, arrays, model weights and
restricted external source bodies are excluded.

## Shared storage and owner handoff

The latest shared reserve has only three unassigned inodes after the standing
376-file obligation. Blindly adding account and HF lock files would be unsafe.
The existing canonical account transient envelope already reserves eight inodes.
This mirror reuses four: account claim directory/owner plus the canonical HF lock
directory/owner. It creates **zero permanent shared files and no additional shared
byte reservation**. The standing byte obligation is 122 MiB.

The local preflight requires the actual pointwise continuation's completed or
stopped receipt, its PID absent, and no matching live account owner/transport.
This is necessary because the frozen healthy account owner checks its entire
376-file future bound before looking at the claim; merely holding the claim can
make it fail instead of wait. Root must serialize this mirror after that owner
ends and before starting another account owner. The remote worker acquires the
canonical account claim, then the unchanged pinned HF lock implementation with
a bounded 30-second wait. Only four owned, distinct, current inodes may retire
four entries from the future count: **376 → 372**, after these four are already
charged to live quota. Fresh quota must still retain 5,000 files and 20 GiB after
all remaining obligations. Stale claims or uncertainty require read-only review;
no cleanup of another owner's locks or automatic retry is permitted.

Only **4 MiB new node-local space** is reserved on evc33. Existing 62 GiB +594 MiB
other liabilities, 4 GiB source scratch, unpaid portions of the 320-GiB raw and
96-GiB context caps, and the 20-GiB node floor remain. Paid cache files are counted
once by device/inode; no originals are deleted. Source/mirror journals/HF caches
stay node-local, and the one-CPU zero-GPU `srun` has no shared output files.

After root freezes the source, `prepare.py --source-git <full-pin>` selects only
committed allowed paths and writes a local manifest. The distinct
`publish_once.py --source-git <same-pin> --execute-reviewed-publication` is the
single explicit publication entry point, **not authorized to run by this README**.
It streams the ignored credential through stdin, requires both locks and a
private repository, makes one HF commit under a unique prefix, then independently
re-downloads every object and checks size/SHA. The manifest/receipt of this new
publication cannot be evidence of its own completion or retroactive mirror scope.
