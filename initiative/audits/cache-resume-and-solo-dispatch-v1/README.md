# Cache-prefix verification and solo pilot dispatch

Root stopped all agents at the user's request and took sole operational ownership.
The failed evc44 pilot is preserved separately. Pilot 839321 is a single fresh
bounded run on previously verified evc37: two H100s, four CPUs, 32 GiB, 30 minutes.
All dispatch prerequisites passed. It was pending at preservation; submission
is not evidence of training. Do not reroute it to unverified nodes just to avoid
waiting. Inspect live Slurm state before any replacement.

The interrupted full cache contains 71,862,845,440 bytes. A complete deterministic
source replay checked every accepted NPZ and JSON hash (48,772 files) and matched
the stable remote prefix. The complete archive identity is recorded in plan.json.
No local full archive was created. No remote append occurred in this first plan.
A read-only account process scan encountered privileged SSH/PAM descriptors and
stopped; the narrowly corrected receiver recognizes only those root-owned session
helpers and retains strict rejection for arbitrary uninspectable processes.
A fresh v2 plan rechecks the full bytes before continuation with the corrected
receiver. Resume outcome remains pending and must be checked from actual receipts.

The new continuation has 23 local regression tests covering arbitrary byte offsets,
no truncation, prefix/source mismatches, interruption, stale identities, overlong
streams, publication refusal and process ownership checks. Neither these tests nor
the prefix proof establish a published full archive. The remote final receipt and
matching completed.json are still required. The accepted masks/folds remain unchanged.

The v2 full replay also passed with exactly the same complete archive and prefix
hashes. Its accepted plan and precise process-scan exclusions are preserved.
All 964 repository tests, Ruff, mypy, archive and v2-coverage checks passed.
The append is dispatched separately; its completion is not asserted here.
