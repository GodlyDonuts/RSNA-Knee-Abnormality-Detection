# Parallel execution option: move C0 to a second ordinary H100 node

Read-only scheduler observation: **2026-10-20 11:27 UTC**. No allocations,
transfers, dispatches or controller changes were made. Existing owner **59538**
remains active and owns all three arms.

## Observed capacity

The healthy ordinary H100 nodes were fully allocated. In particular, evc27,
evc33 and evc48 each had both GPUs assigned. evc38 showed one unused GPU but has
unresolved prior CUDA failures; evc26 was invalid/draining and evc32/37/50 were
under maintenance. None is an immediately usable third GPU.

Teacher job **975297** was still running on evc48 with approximately 3 hours
33 minutes left before its wall limit; actual completion can occur earlier.
Its release would make evc48 a candidate, subject to other queued work. The
visible account/QoS fields did not show a numeric GPU cap, which is not a
guarantee of allocation. Slurm's displayed timestamps are EDT, not UTC.

## Potential benefit

Keep **B/C on evc33** and consider **C0 on evc48**. The original 144-exposure
engineering runs imply roughly **5.3–5.4 hours per arm** when scaled to 23,322
study exposures. Their fit timer already includes CPU context loading.
Three arms on two GPUs therefore require roughly two waves; a third GPU could
save approximately one five-hour wave. These are estimates from a small
engineering cohort, not guaranteed production timings or queue forecasts.

The complete context cache remains a shared prerequisite. At the 11:20 UTC
observation, 464 contexts occupied 8,126,255,514 bytes, projecting about 68.1 GB
for 3,887 studies; the accepted bound remains 96 GiB. Moving that projected
cache would take about 11–22 minutes at 100–50 MiB/s, or 108 minutes at
10 MiB/s. Those transfer rates have not been measured. No raw DICOM or
500 GB source replica is needed for this route.

## Requirements before changing placement

1. **Transfer C0 ownership before any original C0 dispatch intent exists.**
   Inspect the live owner's journal and remote dispatch state, then use one
   explicit handoff that prevents owner 59538 from also submitting C0 on evc33.
   An independently launched second owner would duplicate the arm. Until that
   handoff is implemented and reconciled, preserve the existing owner.
2. Confirm the original complete context terminal and all 3,887 content/roster
   bindings. Copy only the exact immutable context NPZs and necessary small
   targets, source, license and provenance files. Destination admission must
   rebind paths/inodes/stat proofs while preserving original content hashes,
   target bytes, fitting order and source context-index identity.
3. Check evc48's actual free space and outstanding liabilities before transfer.
   Budget up to **96 GiB context cache + 12 GiB C0 arm storage + 64 MiB small
   controls**, preserving **20 GiB free plus every other existing liability**.
   Destination capacity has not been freshly observed. Use node-local controls
   and logs; no new shared large files or generous shared-file allowance is
   assumed.
4. Reuse evc48's existing parent/runtime only after exact source, weight, license,
   version and stat checks. Required runtime is torch 2.14.0+cu130, timm 1.0.30,
   CUDA 13.0. Reconstruct the matched initialization and require common tensor
   SHA `dbf69ff87261f7a0657c866ef8e41db8b125a62c6245d80017945c634e60158b`.
   A missing or incompatible runtime needs separate funding and admission.
5. Keep scientific source `b361ee40939d875d3adf82d5e04dbf2d7d62b0a0`, the worker,
   publisher, targets, recipe and all numerical modules unchanged. Only
   placement/admission/ownership controls need a distinct, explicit binding.
   Request one H100 and two CPUs on an ordinary one-/two-GPU node after inputs
   and funding are ready; no four-/eight-GPU node or healthy-job displacement.

This option is worthwhile if transfer and queue delay stay below the avoided
second training wave. It does not accelerate the unfinished common context
preparation, and it is not authorization evidence that a handoff or transfer
has already occurred.
