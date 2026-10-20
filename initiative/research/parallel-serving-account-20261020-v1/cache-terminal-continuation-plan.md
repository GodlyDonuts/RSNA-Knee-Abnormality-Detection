# Contingency for a terminal cache wall limit

Prepared October 20, 2026; local review only. No new owner, allocation, transfer,
cache, training run or admission has been created.

CPU 980341 at 13:32 UTC observed raw 842/context 840 of 3,887. The raw job has produced
320 new studies in 5,677.53 seconds: about 3.38 studies/minute, versus 2.804 needed to
finish its original 3,365 missing studies within 20 hours. The last interval was
about 3.53/minute. This currently leaves about 3.4 hours of wall-limit slack at the
cumulative pace. Leave 979664/979665 and owner 68411 unchanged.

## Smallest continuation

Use one distinct, finite local controller and retain the existing source-resume,
one-case decode/parity and context-write functions. This is a one-generation
successor, not a retry loop or a fresh acquisition. Require exact root-job
`sacct` terminal evidence and an account-wide `squeue` filtered to the exact IDs.
Observation loss, a slow interval or an estimated finish time never authorizes it.

| Actual condition | Operation |
| --- | --- |
| Both original v3 producers complete 0:0 | Collect the originals only. |
| Raw 979664 complete 0:0; context 979665 TIMEOUT and absent from queue | Continue missing contexts only; no source download. |
| Raw 979664 TIMEOUT and absent from queue | One missing-only raw successor. Keep any still-running context 979665 untouched. |
| Context 979665 subsequently TIMEOUT, or FAILED specifically because its bound raw 979664 timed out | One missing-only context successor after its own terminal/empty-queue proof; bind the new raw producer. |
| Unrelated failure, conflicting proof, existing uncertain new intent | Preserve and reconcile; no automatic redispatch. |

If a TIMEOUT nevertheless left a complete terminal/index, reconcile all 3,887
original proofs and emit an honestly TIMEOUT-bound sealing receipt; do not
fabricate COMPLETE 0:0 or rebuild the data. A small distinct sealing job may
produce a new COMPLETE receipt without acquiring or rendering any case.

## Necessary worker deltas

- Inventory an explicit finite set of roots: initial 16, original failed full
  cache, v3 and the one new successor. Keep source roster, revision, corrected
  cases.py and all numerical modules pinned. Remove the historical assertion
  that the *current* reusable prefix is exactly 522; retain 522 as its ancestor
  receipt, then calculate the current complete/pending union.
- Separate proof ownership from NPZ ownership. `reconcile.inventory` currently
  assumes a proof and its cache live in one root. V3 legitimately records an
  intact orphan retained under the older root. Reuse it only through its exact
  path/stat/SHA/source proof, not by changing the old row or copying the NPZ.
  Duplicate identical references may merge; conflicting rows must fail.
- Accept both `fitting-study-*` and the source-resume helper's
  `fitting-resume-*` scratch layout with their exact metadata/source-table
  binding. The current inventory recognizes only the first name. Preserve
  original partial JSON, NPZ and source buffers. Resume only missing source
  bytes; hash/verify retained partial files as the existing helper requires.
- Reuse completed raw/context files by their existing whole-byte proof and
  current stat. Re-admit only unsealed orphans against retained original pixels
  or exact current rendering; otherwise write a distinct successor file.
  Context rows keep their actual original/corrected renderer provenance.
- Count every generation, orphan and source buffer by unique inode against the
  existing 320 GiB raw/96 GiB context/256 MiB raw-control/32 MiB context-control caps.
  The current 4 GiB scratch calculation covers only one predecessor: inspect
  retained physical bytes and select one/two lanes that actually fit before
  dispatch; never delete originals or assume another two buffers fit.
- Keep original ancestry fields and add a bounded `continuation_chain` binding
  immediate predecessor job/root/intent/source/actual terminal, aggregate
  reused/new/orphan counts, and the new producer's exact root/source/index SHA.
  The final acceptance status can remain the existing alias-recovery family:
  it still describes the same corrected inputs, with extra truthful ancestry.
  Reuse `collect.py.txt` unchanged for an actually COMPLETE successor; it already
  accepts explicit root/job/source and checks all 3,887 current file stat proofs.

## Consumers that must explicitly accept the new roots

The live fitting owner cannot consume new receipts merely by finding new files.
Its frozen `owner.py:cache_receipts` uses the v3 audit directory;
`owner_remote.py:cache_roots/remaining_cache` map only the v3 roots; and
`admit_cpu.py:admit` hardcodes those roots and their immediate intent chain.

Before any changed consumer starts, reconcile the existing fitting/deployment
owners as still idle with no admission/dispatch/phase intent, and hand off their
sole ownership. Use a distinct source-pinned operational adapter/owner: change
only collection-location selection and the cache endpoint/terminal-chain
validation, preserving the admitted numerical worker, recipes, targets, base,
initialization and publication source. Bind the adapter's actual source/SHA in
its admission evidence; never present adapted admission code as the old bytes.
Its final 3,887 UID join, renderer counts, index/hash/stat checks and funding
checks remain unchanged. Do not overwrite prepared controls or old receipts.

The optional portable owner also calls the frozen v2 `cache_receipts`, and its
derived remote control inherits the same maps/admission. It needs the same
explicit cache adapter. `placement.check_rows` and context transfer do not need
different numerical logic when final status/schema and 3,887 ordered rows remain
unchanged; preserve the new terminal/index/chain in the source capsule. Rebind
any waiting deployment owner to the actual final receipt location/placement.
No model retraining, renderer reacceptance or new label/cohort construction is
needed merely because completed cache bytes have more than one producer.

## Bounded checks before that one dispatch

Reuse focused tests with: TIMEOUT versus RUNNING/unknown/queued; context-only
continuation; raw-timeout-induced context failure; mixed-root completed orphan;
partial JSON/ZIP and `fitting-resume-*`; exact CRLF metadata bytes; duplicate or
changed proofs; old/new renderer lineage; original-file byte/stat preservation;
completed-union collection; and a full consumer adapter dry run on synthetic
proofs. No broad training retest is warranted for unchanged numerical modules.

Fresh quota must retain the then-current model/deployment obligations (currently
123 MiB/376 shared files, plus node obligations), including any actually added
control/log files. Reuse a serialized funded transient envelope only with an
explicit accounting proof. The successor's finite duration must leave time for
fixed-final fitting and qualification before the existing October 22 noon UTC
launch deadline. No new human approval is required by this plan; no dispatch is
authorized by this document alone while the original jobs remain healthy.
