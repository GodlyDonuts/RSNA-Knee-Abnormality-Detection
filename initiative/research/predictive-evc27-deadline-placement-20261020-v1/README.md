# All three predictive arms on evc27

**Actual October21 06:02UTC:** all3,887 inputs and the one67,005,012,716-byte transfer are accepted. **C989035 COMPLETED0:0 /5:14:41**, all six epochs/2,916 updates/23,322 exposures. **B989036 is finishing epoch6; C0 989037 RUNNING since05:44:57UTC.** Both have finite real progress and empty stderr.

C1–6 and B1–5 are privately preserved exactly once. C6 publisher993470 completed0:0 /28seconds; all five objects were independently downloaded/size/SHA verified at HF `72b7064e4a3eb06a1bc4ac6b9dae686eaa7ed779`. Model SHA `8231ea03a4109509ec162e2a19b0a477de1affe31ac4ac53ab22bf287723bfbd`. The original return failed on file ordering; [the reconciled correction](../predictive-final-return-order-recovery-20261021-v1/README.md) has now independently returned all seven original C objects to evc33, preserving the failure. No training or publication was repeated. B/C0 final weights and returns remain prerequisites for combined trained T4 qualification; no predictive score exists. [C terminal](actual-fixed-final-C.json); [preservation review](actual-C-epoch006-preservation-reviewed.json); [accepted return](actual-owner-return-C.json).

| Preserved checkpoint | Original private revision | Original receipt |
| --- | --- | --- |
| C1 | `2c18376d34d850a401d0417af904df095be149bc` | [C1](actual-owner-publish-C-001.json) |
| C2 | `42375a6af4d47a5d34f64bcc9205dda5b6429192` | [C2](actual-owner-publish-C-002.json) |
| C3 | `89e9f2f757df92ca710efec7c525bdb48d099c9c` | [C3](actual-owner-publish-C-003.json) |
| C4 | `a7c9fb3ef9ae93de9687680640816f4161c8764b` | [C4](actual-owner-publish-C-004.json) |
| C5 | `0c845d35ec7c8d2d8767013d936a3615d06f9a61` | [C5](actual-owner-publish-C-005.json) |
| C6 | `72b7064e4a3eb06a1bc4ac6b9dae686eaa7ed779` | [C6](actual-owner-publish-C-006.json) |
| B1 | `16ca1e622d1ee169c8f1c9ce3109c9c148f234da` | [B1](actual-owner-publish-B-001.json) |
| B2 | `ae967658f325fbcfb2e16a44ebe0430b0307ce55` | [B2](actual-owner-publish-B-002.json) |
| B3 | `2a9e1b4b38dda7676eeccb35dc3722313f5b13f3` | [B3](actual-owner-publish-B-003.json) |
| B4 | `c77d3106dad5e4ceb1e40d1d428054fa7e0b1973` | [B4](actual-owner-publish-B-004.json) |
| B5 | `bdffd94a710266766de777d6eb4f94fb33746554` | [B5](actual-owner-publish-B-005.json) |

C0 actually started05:44:57UTC, replacing the old queue estimate. Its eight-hour allocation ends13:44:57UTC; actual training throughput determines completion. No optional full run or ready-subset deployment is admitted.

**Resolved monitoring failures:** Slurm created each newly allocated job’s stdout/stderr as0644. The frozen controller requires0600 and reported observation errors for all three arms. Root narrowed C’s exact owned pair at00:37UTC, then B’s pair at00:47UTC after B actually allocated. Only permissions changed; no body writes, source/input changes or training restart. The existing controller emitted all three recovered receipts after each correction. [C correction](actual-C-log-permission-correction.json); [B correction](actual-B-log-permission-correction.json); [second C recovery](actual-observation-C-recovered-002.json); [second B recovery](actual-observation-B-recovered-002.json); [second C0 recovery](actual-observation-C0-recovered-002.json). C0's pair was narrowed at05:45:43UTC after actual allocation, before a third observation incident was emitted. All three original log pairs are now0600; do not repeat. [C0 correction](actual-C0-log-permission-correction.json).


Authenticated Kaggle at00:44UTC reports **five available entries** and **3,587.5076641 free GPU-VM seconds**. The existing3,360-second primary qualification/public-execution reservation remains protected. The remaining prerequisites are B/C0 finals and combined trained qualification, not daily entry allowance. [Account receipt](actual-account-20261021-readonly.json). Existing owners33107/32378 handle observation, private epochs and deployment; do not add or restart a controller.


At **October 20 20:15:31 UTC**, fitting continuation **33107** and deployment **32378** were live with empty stderr, waiting for the complete inputs. The [actual live receipt](../predictive-evc27-startup-continuation-20261020-v1/actual-live.json) binds the unchanged `2a224ae0` controls. No full-cache transfer, GPU fit or account phase had started at that historical observation; the actual current transition is above. Original fitting32377 stopped before any phase because its launcher marker triggered the startup guard; the distinct continuation preserves that failure and the original journal.

The actual 19:55:26 UTC scheduler observation placed both evc33 H100 job end times after the intended entry window. This successor moves B and C to the already planned C0 destination, evc27. The preceding idle fitting/deployment/observer owners were stopped with explicit absence proofs; no remote job was cancelled.

The numerical source remains `44cc559e947560a635c34fee7b1f2a3b7be3a241`, with the original CPU preparation 979667, parent, shared initialization, targets, corrected-source lineage and six-epoch recipes. The original portable initializer validates **all three models** against that preparation. B, C and C0 receive separate recipe, input, funding and binding files; the worker and publisher bytes are unchanged.

The sole fitting controller first admits the complete 3,887-case source on evc33 for CPU validation. It then streams the exact contexts to evc27 once, verifies every size and SHA, and shares that destination index and existing parent/runtime among all three arms. It submits C, B and C0 in that order, each requesting one H100, two CPUs and eight hours. The physical two-GPU node permits at most two simultaneous fits. No GPU is reserved while inputs are incomplete.

The two-wave worst-case training allowance is 16 hours, excluding uncertain queue delay. Initial admission must occur by October 21 09:15 UTC, transfer must start by 10:45 UTC, and dispatch by 14:00 UTC. The controller preserves original results until October 22 noon UTC. These cutoffs are planning limits, not scheduler guarantees.

## Storage and retained obligations

Fresh CPU observation 983717 at 19:57:47 UTC found 831,252,013,056 bytes free on evc27. The accepted complete scope requires 394,815,078,400 bytes including the standing floor, leaving 436,436,934,656 bytes beyond those obligations. The actual receipt is `actual-funding-capacity.json`.

| Obligation | Reserved scope |
| --- | ---: |
| Original evc27 obligations | 191 GiB + 564 MiB |
| One complete context copy | 96 GiB |
| Original and additional controls | 64 MiB + 64 MiB |
| Three primary fitting/publication outputs | 3 × 12 GiB |
| Original C100 pilot | 6 GiB + 4 MiB |
| Prepared full C100 | 12 GiB + 16 MiB |
| Prepared generative-routing pilot | 6 GiB + 4 MiB |
| Descriptor-normalizer artifact | 1 MiB |
| Standing free-space floor | 20 GiB |

Optional work receives no speculative paid credit. Original evc33 obligations remain reserved. Shared storage retains 123 MiB and 376 future files; the same six original GPU log paths are reassigned, with a combined 8 MiB bound per arm. Only exact job-bound, owned files already present in the live quota can retire their corresponding file obligations.

## Preservation and final returns

The controller observes each arm independently and uses the original publisher for all six epochs per successful arm. A malformed observation or unresolved publication on one arm does not authorize a restart or prevent the other arms from being preserved. If dispatch becomes ambiguous, `dispatch_reconcile` reads only original sealed submission, script and binding proofs; it never retries `sbatch` or invents a job from an unsealed response.

Each successful fixed final returns exactly seven original objects to a distinct evc33 mirror: terminal receipt, independently verified epoch-006 publication receipt, model-only weights, provenance, source manifest, object manifest and license. The sender checks the actual completed GPU job, binding and six original verified revisions. The receiver checks the expected arm/job/binding before copying, then independently verifies every length, SHA and stream EOF. Each return is below 512 MiB; no optimizer, MRI, context or target copy is included. The three mirrors fit within the retained evc33 obligations.

Returns retain the existing 30-minute receiver accept timeout, 10-minute connected-stream timeout and 45-minute CPU allocation. A timeout preserves partial output and stops that mutation path for reconciliation; it does not automatically retransmit. The canonical deployment adapter uses the returned model/provenance/license paths, validates all three original source/input proofs, and retains the trained qualification and submission gates. No accuracy or runtime acceptance follows from this placement change.

Local remote tests execute the frozen dispatcher without a scheduler, confirm the three original node27 job scripts and exact-once behavior, exercise complete and partial dispatch reconciliation, test one-index admission with separate original recipes, check funding and paid-log boundaries, and verify arm-specific final-return derivation. They do not constitute actual input admission or training.
