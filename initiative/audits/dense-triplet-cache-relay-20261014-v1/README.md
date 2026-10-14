# Private once-only public-triplet cache CPU relay evc31 -> evc40 (PREPARED, not dispatched)

Issue 78. `scripts/newton/triplet_cache_relay.py` (`public_triplet_cache_private_relay_v1`)
prestages ONLY the immutable public ordered-triplet cache the live dense fit needs, at the
identical absolute path `/tmp/rsna-kaggle-public-source-shards-20261013-v1/merged-cache` on
evc40. Devin performed no transfer, dispatch, funding or remote mutation. Original 899766/908454
training, science, trainer checkpoint/root/journals are unchanged; those wait for the actual
dense exit-75 terminal and the PR76 transition record.

Private manifest: derived from the original pinned `public-ordered-triplet-v1/index.json`
(SHA 849855…, cohort 6d00d9aa…, fitting/evaluation roster hashes bound by actual CPU receipt
158c242f…): the index bytes first, then every expected `members/<uid>.npz` (3403 × 8130228 =
27667165884 bytes, root's measured member bytes). The sibling merge receipt
`merged-cache-verified.json` (d2c0ad74…) must bind that index/3403 studies/20 shards. Producer
bookkeeping (`intent.json`, `finalized.json`, `receipts/`) is counted in the plan aggregate but
not relayed; root's PR76 `transport.json` must describe exactly the destination tree.

Root-only order (one CPU / `--gres=none` per endpoint, no GPU funded or reserved):

1. `plan` inside a one-CPU zero-GPU step overlapping 899766 on evc31: read-only checks of
   merge receipt, index, exact NPZ directory membership, regular single-link non-symlinked
   files and sizes; writes the private per-member manifest/signatures exclusively to node-local
   `/tmp/rsna-dense-triplet-cache-relay-20261014-v1/source-plan.json` and prints aggregates only.
2. Root separately admits `funding.json` in the same evc40 control directory: status
   `root_admitted_node_local_triplet_cache_relay_funding`, cache bytes (index + NPZ), control
   bytes ≤64MiB, reserve ≥20GiB, all existing target liabilities retained,
   `required_free_bytes` = liabilities + cache + control + reserve, zero shared growth, zero
   deletion credit, zero GPUs, `GPU_reserved: false`.
3. `pipeline()` builds the single `source | target` Newton-login command (stdin/RAM only, no
   raw bytes through the Mac). Source rechecks every planned stat signature before and while
   streaming via the tested `native_cache_relay.source_stream` framing. Target requires an
   allocated evc40 Slurm step with no GPU, the exact funding SHA, absent cache/intent/receipt,
   writes one exclusive intent, verifies header SHA/count/funded cap, the index against its
   pinned SHA before any NPZ, every member size/SHA with `xb` writes and fsync, reserve during
   streaming, trailer/EOF and the exact relative layout, then writes `target-verified.json`
   exclusively (aggregate counts/hashes only).

No retry or overwrite. An uncertain transfer is examined only with read-only `reconcile` after
both original handles stop; it reports an aggregate verified prefix and never deletes.

Tests: `tests/test_triplet_cache_relay.py` (34) plus unchanged `tests/test_native_cache_relay.py`.
