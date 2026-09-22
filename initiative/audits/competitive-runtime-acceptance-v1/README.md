# Competitive runtime acceptance

These are actual CPU checks using the complete accepted release before selecting
16 diagnostic training and 16 outer-fold studies. No gold or reserved confirmation
predictions are generated. Tiny random encoders establish software operability,
not DINOv3 performance. Each acceptance JSON binds its exact runtime and config.

`competitive-cpu-acceptance-attempt002` passed an earlier runtime.
`competitive-cpu-acceptance-cadence-v2` passed the current scheduled-snapshot and
checkpoint-only-capable runtime: two epochs, exact full checkpoint equality after
a within-epoch interruption, identical outer predictions, an exact diagnostic
epoch-2 snapshot marked ineligible for submission, and three public predictions.

Two other attempts hit the enforced disk reserve. Their failure logs remain,
alongside receipts for removing reproducible caches and a redundant clean recovery
checkout. All source data, Git objects and immutable remote archives were retained.
Portable logs replace machine-specific path prefixes; original hashes and untouched
private logs retain the distinction. GPU/NCCL and Kaggle inference are separate checks.

Two-process Gloo acceptance remains incomplete locally. Attempt v1 stalled in
macOS standalone IPv6 rendezvous and was deliberately stopped. Explicit IPv4
resolved that issue in v2; the first real optimizer step succeeded, but OS swap
reduced free space below the checkpoint reserve. Both original failures are
preserved. The bounded helper now uses IPv4 rendezvous and cleans up a process
group on timeout. Complete multi-process recovery on Newton with adequate space.

After the single-process acceptance, inference counter access changed from
`dict.get` to required-key indexing after exact-type validation to satisfy mypy.
The architecture and training algorithm are unchanged; focused inference checks
cover the final source separately. Do not claim byte-identical source between
the earlier acceptance and this final type-checking correction.

## Completed successor: two-process CPU/Gloo

After verified storage reclamation and transient OS memory pressure settled,
`competitive-gloo-acceptance-cadence-v3` passed in 173.99 seconds. Both ranks ran
the real accepted-release diagnostic; a within-epoch stop/resume reproduced the
entire checkpoint tree and outer predictions exactly. The scheduled diagnostic
snapshot stayed ineligible for submission and public-example inference passed.
The receipt's runtime and helper hashes match their final local source bytes.
This supersedes the earlier local-incomplete status; all failed attempts remain
preserved. Actual CUDA/NCCL recovery and pretrained H100 performance still need
the Newton pilot.
