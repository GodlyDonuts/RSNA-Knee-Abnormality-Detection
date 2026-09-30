# Whole-series original-base delta deployment acceptance

The live whole-series control uses scientific source
`c9fcc7eefb09103942be89bf4fa87410eb9ec5ab`. Its core `rsna` runtime remains
unchanged. The external Kaggle deployment wrappers now preserve all accepted
series when the training configuration declares `series_sampling=all`, bind the
saved sampling contract, and reconstruct schema-2 learned tensors against the
exact bundled official original weights. The previous wrapper imposed a six-series
cap and understood only self-contained full checkpoints.

CPU regressions verify full and delta storage give identical direct/chunked MRI
predictions for both no-context and neighboring-context configurations; changed
original base tensors, paths, hashes, storage and sampling contracts are rejected.
Nine healthy series remain available despite the old six-series setting.
Completed production snapshots remain required by the competition coordinator;
diagnostic models never enter that ledger or its submission path.

A private offline T4 engineering probe is being prepared using the separately
preserved 16-study/four-update diagnostic from job 865346, SHA
`5d1b3b562160e803442d33316779401a4fa4721370b182187802b14c78a1c22e`.
It must prove original-base/delta versus expanded-full FP32 prediction and CSV-byte
equality on the three public MRI studies and three previously profiled largest
training-image workloads, without labels or a competition submission. It is not
staged or dispatched at this source milestone and is not an accuracy result.
Production deployment is not accepted until the exact private T4 receipt passes.

All-development epoch 4 completed 1,704 updates and its four private objects were
fully re-downloaded/size/SHA verified at HF
`27e8f49faf26bccd12a6d8fb8aa6d7112fc82a8b`. CPU retention 865372 removed only
the redundant, independently preserved epoch-004.pt, retaining its immutable
sidecar, the live recovery checkpoint and private bytes. Its former link count
was one; 1,338,164,780 shared bytes were reclaimed. Do not repeat either publication
or retention. All future staging still requires a fresh projected 20 GiB byte
reserve and file reserve including both live runs and their deployment copies.

At 08:45 UTC September 30 exact A32 epoch-10 submission 56697342 remains
pending or unknown, with null status/error/score and four authoritative live slots.
No new competition submission is made here. The owner permits worthwhile
checkpoints; use genuine evidence and verified deployment, not remaining quota
alone. Diagnostic execution does not consume a competition submission slot.

Full source checks pass: 1,159 tests, Ruff/format, mypy 38 modules, archive hashes
and current v2 89/89 draft/review coverage. Historical v1 completeness remains
false as expected. Fresh projected shared reserve is 21.130859 GiB, including
remaining primary snapshots/live/atomic files, both production deployment pairs,
12 control delta files, this diagnostic ZIP and 132 MiB metadata/source. The
diag payload will hard-link immutable original base/delta instead of duplicating
them; no hypothetical future cleanup is credited.
