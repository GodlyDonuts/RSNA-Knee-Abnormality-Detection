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

## Exact terminal failure and correction

**Actual bounded CPU acceptance and new private execution:** job **865394
COMPLETED 0:0 in 1:08**. Real official-base/delta expansion with a deliberately
limited six-GiB reported scratch budget passed all frozen and learned tensor
checks. Required budget: 4,669,188,465 bytes; exact expanded file:
1,338,961,540 bytes. The temporary reference was removed. Scientific source
`c9fcc7e` is unchanged; external correction is source `1630cc8`.

One distinct private offline T4 **godlydonuts/rsna-delta-t4-fix3-5d1b3b562160
v1 / ID 136514660** is RUNNING with exact source/settings checked. It reuses
the existing fully verified private Dataset **12288274 v1**, without another
Dataset or large ZIP. Separately hashed replacement code is embedded in the
saved bootstrap and confined to owned runtime scratch; original mounted bytes
and shared inference source remain intact. `fix3/` preserves the exact CPU
contract, scripts, returned notebook identity and dispatch/status evidence.
Full-versus-delta FP32 six-study MRI and CSV-byte acceptance is still a gate;
this is not an accuracy claim or a competition submission. Source checks pass
**1,168 tests**, Ruff/format, mypy, archive and v2 complete coverage.

**Later fix2 terminal outcome:** the exact private T4 v1 ended ERROR in 615.4s.
The second trace passed the 415+2 partition checks and then stopped at the
reference-file scratch guard, which required 22 GiB free before writing a
1,338,961,540-byte temporary file. Free space was not logged; no MRI inference
ran. Exact evidence is `fix2/terminal-failure.json` and the visible log. Both
failed private versions remain immutable and no competition slot was consumed.

The next helper uses a measured reference upper bound (model tensors plus delta
envelope and 64 MiB metadata), a one-GiB bounded MRI chunk and two-GiB temporary
safety headroom. It logs actual/required bytes, refuses insufficient space and
existing paths, and checks actual serialization against its bound. Newton and
Mac reserve policies are unchanged. The exact replacement source is embedded
and separately hashed in a distinct private diagnostic notebook; shared inference
source must match the preserved Dataset manifest. It copies only small owned
runtime code and leaves the already fully verified original Dataset bytes intact.
Private role checks forbid this route for competition-eligible bundles.
Real-checkpoint CPU expansion and actual T4 full/delta six-study MRI equivalence
are still required before accepting this route.

The exact small scored-result, correction, CPU proof and private T4 dispatch
milestone is on GitHub main **3275e1f** and privately mirrored at HF
**726f8691785403af71340b1583a04ea768ffd8e7**. All **163 objects** were fully
re-downloaded/size/SHA verified. This does not imply terminal T4 MRI acceptance;
that distinct private run remains in progress.

CPU-only corrected preparation **865385 completed 0:0 in 1:55**. Using the
actual verified official base and diagnostic delta, it reproduced the old
helper's error, expanded the corrected helper, and checked all frozen and learned
values exactly against the checkpoint. The 1,338,961,540-byte full reference
existed only in owned `/tmp` scratch and was removed; no shared full copy was
made. Exact helper/projection/CPU receipts are archived in `fix2/`.

Distinct corrected private Dataset **godlydonuts/rsna-delta-base-probe-fix2-5d1b3b562160**
v1 / ID **12288274** was created once and all 49 payload objects were fully
downloaded/size/SHA verified. Distinct private offline T4 notebook
**godlydonuts/rsna-delta-t4-fix2-5d1b3b562160 v1** was pushed once, with title
and slug aligned and returned identity checked. Successful actual T4 MRI
equivalence remains outstanding; neither private diagnostic is competition
eligible. The exact preparation ZIP is 1,339,497,581 bytes,
SHA `c385b73ea88e445e1faec0de88473d7609e19d0f56c2961655de044eef1a447d`.
The fresh post-preparation quota projection leaves **21.009223 GiB** and
**7,258 files**, budgeting both production deployment pairs, future snapshots,
live and atomic writes; both existing diagnostic ZIPs count in actual usage.

The actual private v1 later ended ERROR. The preserved visible log identifies
`delta_probe.py:83`: the engineering assertion compared the complete frozen
encoder state count with 415, the count of official backbone tensors alone.
Its strict base/delta restore had already passed. The saved state has 417 frozen
encoder entries: 415 official tensors and two grayscale normalization buffers,
`encoder.mean` and `encoder.std`. The correction checks both parts explicitly,
retaining strict hashes, shapes, dtypes and learned membership. Regression tests
execute expansion and reject changed normalization hashes or backbone membership.
Real-checkpoint CPU expansion and a distinct exact private T4 acceptance remain
outstanding. No MRI inference ran in v1 and no competition slot was spent. This
does not identify the cause of earlier hidden competition errors.

`private-probe-v1-terminal.json` binds the actual private identity/source and
`private-probe-v1-terminal.log.txt` preserves the visible terminal trace. Never
retry the old create/push or call this notebook accepted. Completed all-data
epoch 5 and control epoch 1 were separately privately published and all four
objects per epoch fully re-downloaded/size/SHA verified; see their distinct
epoch-publication audits.

## Actual private diagnostic dispatch

CPU preparation 865374 COMPLETED 0:0 in 1:21 with exact original-base and
scientific-source verification; stderr empty. The 49-object ZIP is 1,339,497,052
bytes, SHA `ac73b7ffbdc501657179b679ebc7414790e85153d2a304458264865c0ec786fd`;
manifest SHA `5fed97851e14ede738d3f18c811eb50b837f51976741171076b460251a0b3ea5`.
Packaging projection `ec798167f47786d8ffa4105a5e97ee4812425e28` retains the entire
unchanged scientific `c9fcc7e` runtime. The diagnostic checkpoint and original
weights are hard-linked to immutable retained inputs; no expanded full checkpoint
was created on shared storage. Fresh pre-preparation projected reserve was
21.129356 GiB inclusive of the one diagnostic ZIP and both production bundles.

Private Dataset `godlydonuts/rsna-delta-base-probe-5d1b3b562160`, ID **12287425**,
version 1 was created once, and all 49 payload objects were fully downloaded and
size/SHA verified. Its attached exact source is SHA
`cfc7920ed1e1ac43d7cbe3ce199deb929361cef014ff84d6098c088a25febc5a`.
SaveKernel was called once and returned version 1 / ID **136505806**. Kaggle derived
the actual notebook slug from the supplied title:
`godlydonuts/rsna-whole-series-exact-delta-t4-diagnostic`. A first read against the
original requested slug returned 403. Read-only reconciliation of the returned
identity verified exact source, private/offline T4 settings, both private Dataset
inputs and official MRI input; **no push was retried**. At 09:08 UTC actual v1 was
RUNNING. Do not use the originally requested absent slug or repeat any creation.

Use the archived `read_probe.py.txt` / owned local helper to inspect the actual
returned identity read-only. Only successful exact terminal acceptance permits
the separate whole-series deployment route. This probe remains ineligible for
competition, and no new submission slot has been consumed.

The separately verified small source/epoch-4/launch milestone is private HF
`1d05f5604c863aede26f22dd5338239245283675`, all 102 objects fully re-downloaded
and size/SHA checked, bound to GitHub main `360c99b`. This mirror does not
represent successful T4 terminal acceptance or a competitive accuracy result.
