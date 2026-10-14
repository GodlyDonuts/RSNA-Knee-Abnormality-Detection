# Private once-only dense state/runtime CPU relay evc31 -> evc40 (PREPARED, not dispatched)

Issue 80. `scripts/newton/dense_state_relay.py` (`dense_suffix_state_runtime_private_relay_v1`)
relays the remaining node-local PR76 state that the once-only triplet cache relay did not
carry, at IDENTICAL absolute paths on evc40. Devin performed no transfer, dispatch, funding,
GPU scheduling, resume or remote mutation. Original 899766/evc31 state is never deleted or
rewritten; original funding/admission/journals and the numerical trainer are unchanged.

Input: only the ACTUAL PR76 `allocation-transitions-v1/001/record.json` of the original dense
exit-75 terminal (schema `dense-suffix-allocation-transition-v1`, original evc31/899766/funding
0f5a2d17…, `original_state_deleted: false`), given with its root-observed SHA. Absent or
differing record bytes are refused. Every manifest member of the six PR76 node-local classes
(`original_root` incl. immutable epochs and the exit-75 checkpoint, `admission`,
`public_cache`, `overlay`, `runtime`, `recovery` = original GPU-recovery output tree) must be
an absolute normalized path inside its node-local class root; traversal, non-node-local roots,
members below links and transition controls inside the manifest are refused. Shared inputs
stay verified in place by PR76 and are not relayed.

Stream rows: the record bytes, the recorded coordinator, then every remaining node-local file.
Public-cache members already relayed by `triplet_cache_relay` (exact pinned index and member
hashes, manifest 09ea7bc6…, 3404 files / 27668082285 bytes) are NOT retransferred: they must
match the record exactly and, on evc40, the ACTUAL immutable
`/tmp/rsna-dense-triplet-cache-relay-20261014-v1/target-verified.json` (SHA bound by funding)
plus a fresh node-local stat/signature proof (regular, single link, 0400, same device, ctime
not after the receipt). Producer bookkeeping not carried by the triplet relay is streamed.

Root-only order (one CPU / `--gres=none` / zero GPU per endpoint, Newton login stdin/RAM):

1. `plan` in a step overlapping 899766 on evc31: read-only exact source member set/sizes/links,
   exactly `{001/record.json, coordinator.py}` as transition controls, coordinator SHA; private
   rows/signatures go only to node-local `/tmp/rsna-dense-state-relay-20261014-v1/source-plan.json`.
   Stdout is aggregate-only (per-class entries/bytes, stream files/bytes, list digest).
2. Root writes `funding.json` in the evc40 control: status
   `root_admitted_node_local_dense_state_relay_funding`, issue 80, source/destination nodes,
   record SHA/bytes, coordinator SHA, transfer-list SHA, stream files/bytes, triplet receipt SHA,
   envelope ≥ stream bytes, control ≤1GiB, reserve ≥20GiB, existing liabilities > 0,
   `required_free_bytes` = liabilities + envelope + control + reserve, zero shared growth,
   zero deletion credit, `GPUs_funded: 0`, `GPU_reserved: false`.
3. `pipeline()` builds the single `source | target` command. Source rechecks every planned
   signature before/while streaming (tested `native_cache_relay.source_stream` framing:
   magic, list digest, count, byte total, END, EOF). Target, before any write: exact funding,
   triplet receipt and cache proof, header digest/count/cap, record bytes, list rebuilt from
   the record, every existing destination member exact (regular, size, SHA, executability,
   stable signature) and no extra member/control. Then one exclusive intent; new bytes go to
   node-local staging (`xb`, fsync), are SHA verified and only then hard-linked into place
   (never replacing a file); original links are recreated; the complete original member set
   is re-verified; finally PR76's exact `001/transport-receipt.json` and aggregate
   `target-verified.json` are written exclusively.

No automatic retry, overwrite or deletion. After an interruption the existing intent refuses
every repeat. `reconcile` is read-only after BOTH original handles stop and privately records
verified prefix members; a later bounded attempt needs root's separate
`prefix-admission-NN.json` binding that reconciliation, the record and the funding.
Existing verified prefix members are then proven, never rewritten.

Tests: `tests/test_dense_state_relay.py` (66), reusing the PR76 fixture and ending in an
actual PR76 `admit` of the relayed destination. No accuracy claim.
