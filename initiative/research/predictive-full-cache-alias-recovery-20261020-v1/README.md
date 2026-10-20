# Missing-only recovery for the actual alias failure

Prepared locally; **not dispatched**. Original raw job 978318 and context job
978414 ended FAILED 1:0. The earlier timeout-only owner stopped without starting
any successor. The retained raw prefix has 506 newly verified cases plus the
16 original cases; these caches are reused, not fetched again.

The failure arose when a whole acquisition contained only frames already
represented by canonical aliases. The original renderer attempted to assign
metadata using an acquisition rank that was absent from the canonical roster.
Only `cases.py` is corrected: an alias occurrence uses its canonical
representative's metadata. The original canonical roster, original pixels,
descriptor targets, support priors and accepted labels stay unchanged.

This owner requires an independently captured **actual retained-case regression**
and its exact SHA before any intent. It binds the original and corrected
`cases.py` hashes, the original job, unchanged successful-domain contexts and
successful rendering of the retained failed case. The live regression is
separate from local unit tests.

`owner.py --source-git <committed-control> --regression-receipt <actual-json>
--regression-sha256 <exact-sha>` freezes its own code and original numerical
sources. Both remote workers require fresh original FAILED 1:0 accounting and
empty queues. They retain every original cache, source buffer, proof and failed
attempt. Existing new intent or uncertain launch prohibits a retry.

The distinct roots are:

- `/tmp/rsna-predictive-full-raw-alias-recovery-20261020-v1`
- `/tmp/rsna-predictive-full-context-alias-recovery-20261020-v1`

Raw recovery keeps two study lanes, four CPUs and zero GPUs. It reuses verified
cache proofs, independently admits intact orphan caches, and uses the existing
missing-only source helper for retained partial buffers. Complete source files
are reused; only absent files or verified missing byte suffixes are requested.
Context recovery uses one CPU and zero GPUs, waits for the matching actual raw
successor, and reuses verified context archives unchanged. Both have a 20-hour
limit; the finite local owner has at most 30 hours. Read-only Slurm inspection
found normal partition MaxTime unlimited and no normal QoS/account MaxWall.

The current rate suggests roughly 15–16 more hours for the remaining inputs;
this is an estimate, not an observed completion time. A second node would need
new source ownership and transport and is not part of this recovery.

Old context rows keep their **original renderer provenance**. Newly rendered
rows name the corrected renderer. An intact old orphan is retained only after
bitwise comparison with the corrected render, and keeps its original manifest.
The final terminal records both renderer dictionaries and the regression SHA.
Downstream fitting must explicitly accept this mixed but bound provenance;
it must not rewrite old provenance to pretend those bytes were regenerated.

All current node liabilities remain funded: other future 62 GiB + 594 MiB,
unpaid parts of raw 320 GiB and context 96 GiB, original/new source scratch
4 GiB, and the 20 GiB floor. The 121 MiB/376-file shared future reservation
reuses the previous unattempted successor log allowances, retaining the
20 GiB/5000-file floor. No deletion credit or new GPU reservation is taken.

Successful collection retains the original receipt/index hashes and all 3887
existing cache stat bindings without another bulk rehash. New sanitized
receipts are `actual-owner-raw_collect.json` and
`actual-owner-context_collect.json` here. Private per-case data and detailed
source records remain on the owning node.
