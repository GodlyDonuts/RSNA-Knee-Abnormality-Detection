# Rank-64 real acceptance and matched production launch

Recovery pilot **865556 COMPLETED 0:0** on evc31 in 15:58. It used 16
training/16 development outer studies for two epochs/four optimizer updates.
All **120** residual projections changed, **415** original DINOv3 tensors stayed
unchanged, and the **20,447,232** adapter parameters match the verified factory
provenance. Full versus delta storage, interruption after one update and separate
resume passed model/Adam/RNG/progress equality and byte-identical outer
predictions. A strict checkpoint-only real-MRI reload also passed. These are
engineering checks, not a competitive accuracy estimate.

The first allocation, **865551**, failed on evc22 before any model/environment
or optimizer update: the allocated driver check timed out (step exit 124).
Its original dispatch and stdout/stderr are retained. The distinct evc31 recovery
was dispatched only after authoritative terminal/output-absence review and
passed allocated CUDA/NVML/header checks. Neither pilot is competition eligible.

The exact prepared production bytes were staged once and job **865578** was
submitted once on evc31, one H100/two CPUs/32 GiB/16-hour maximum. At the first
read it was RUNNING, had passed allocated device/header/CUDA context checks,
and had empty stderr. The 15:16 UTC observation verifies **47 finite real production optimizer
updates**, with empty stderr; final accuracy is still outstanding. Never duplicate the job or reuse its output directory.

Scientific source is `c9fcc7eefb09103942be89bf4fa87410eb9ec5ab`; the accepted
labels, masks, outer fold and MRI inputs remain unchanged. The sampled-input
comparison keeps six selected series, 16 uniformly sampled slices, 224-pixel
inputs from the 256 cache, effective batch eight, attention head and the same
10 × 281-update schedule. Only encoder adapter rank changes from 32 to 64.
The matched rank-32 reference is **0.8620940794 internal macro AUC** on 1,156
studies and **0.769 public Kaggle**; rank 64 has no accuracy result yet.
Same seed does not guarantee the same head initialization when the adapter
parameter count changes. Repeated folds/seeds are needed for broader rank claims.

The actual terminal all-data primary released its future full-checkpoint write
obligation; its final bundle and ZIP were already materialized and verified.
No future cleanup was credited. The fresh complete production budget leaves
**21.436870 GiB / 6,259 files** after both remaining jobs, future deployment
copies, diagnostic allowance and metadata. See `dispatch-acceptance.json` for
actual quota and each obligation. Parse numeric quota fields independently of
whether Lustre prints the filesystem on the same line.

- Control: `/home/sa305415/rsna-knee/production-launch-adapter-a64-sampled-c9fcc7e-v1`.
- Run: `/home/sa305415/rsna-knee/runs/dinov3-l16-adapter-a64-sampled-fold0-c9fcc7e-production-v1`.
- Pilot: `/home/sa305415/rsna-knee/rank-capacity-a64-c9-pilot-v1`.
- Role-bound publisher SHA: `ca430ec5df834c0dfc2744f24c54ff853eb5a86713ce2aa5738aadfbb5cb740c`.

Use `scripts/hf/dispatch_adapter_a64_epoch.py N --ssh-control SOCKET` only for
new completed immutable snapshots after inspecting existing local/remote intents
and private paths. Every epoch must be uploaded once and fully re-downloaded/
size/SHA verified. The original production-files manifest retains its immutable
preparation-time status; actual staging/launch state is in these separate
receipts. At most two GPU nodes remain authorized. Current other job is whole-
series rank-32 control **865357** on evc28; primary **865263** is COMPLETED 0:0.

The complete milestone is on GitHub main **2f85557**, privately mirrored at
HF **e0090eb0976aa0969f64d6d68cf8dc11a93ad3d3**. All **178 source/evidence/
preview objects** were fully re-downloaded/size/SHA verified. The exact parent,
manifest and object map are in `hf-completed-fit-*.json`. Never repeat its
commit. This small mirror does not publish the ~147-MB derived native volumes.

Production E1 / **281 updates** is now complete and privately preserved once,
all four objects fully re-downloaded/size/SHA verified at
**1d76bece2114c6770bd889959f4212443bde2e83**. The actual production role
contract passed before publication. Do not repeat E1; inspect existing intents
before each new epoch. Exact snapshot/journals are in `epoch-publication/epoch-001/`.
The same job entered E2 with finite updates and empty stderr. No rank-64
held-out result exists yet.

Production E2 / **562 updates** and E3 / **843 updates** have now also passed
the exact production snapshot contract. Each epoch's four private objects
were fully re-downloaded/size/SHA verified once: E2 at
**dd7a323eaf3ffdbf429febb070789f0f7ef618b5**, E3 at
**06618b46ce73bec9ec254b8b3bc570bb1c39b6cb**. E3 checkpoint SHA is
`0120b4cd0efa21e0718ded9ab1d7b6a28ad1214b671603ec7ca5922689c3e030`.
Exact immutable sidecars, local dispatch records and remote journals are in
`epoch-publication/epoch-002/` and `epoch-publication/epoch-003/`.
Do not repeat E1–E3. The same job continues; no final held-out accuracy is
available yet. The new source-based native384 caches are engineering artifacts
for a separate future comparison, not inputs to this rank-64 run.
