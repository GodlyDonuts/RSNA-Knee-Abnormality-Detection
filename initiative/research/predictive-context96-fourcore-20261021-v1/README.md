# Dense96 four-core serving comparison

Prepared October 21, 2026. This is a distinct successor to the held dense96
public notebook; the original remains unchanged and unsubmitted.

The measured failure to address is runtime: its last public preparation projected
9.44 hours at 1,300 studies/two lanes with setup and tail allowance. The separate
actual CPU thread experiment found 19.95% less warm preparation with two intra-op
threads per lane and <=1.1921e-7 image differences. That does not establish a
trained numerical comparison or a cold runtime gain.

This source preserves the exact C100 final model, all 96 encoder centers per
context, complete descriptors/supports, original decoder and fitting provenance.
On the exact three-study public roster it prepares the two-thread candidate
**first**, then a one-thread reference after joining the first pool. It compares
all context fields, actual trained FP32 and FP16 predictions, and cross-device
predictions. Actual private paired arrays are retained in Kaggle outputs and
independently checked in owning-node RAM. No images or individual predictions
are collected to the Mac/Git. Hidden execution has two threads per lane and no
reference pass. Public qualification keeps strict memory/precision limits.

The old diagnostic only saved hashes and aggregate differences, so its exact
prediction discrepancy cannot be recovered from those files. This new paired
comparison does not invent an explanation or retroactively accept that version.

The public watchdog is 480 seconds, plus 120 seconds startup reservation. The
single coordinator requires fresh >=2,520 free GPU-VM seconds, preserving the
original Colab final deployment's 1,920 seconds. It cannot submit a competition
entry. Existing intents are never retried. Runtime acceptance is a separate
measured decision; no accuracy improvement is assumed.

Four focused tests passed: compilation through the real source layers preserves
model/provenance, actual tensor preparation rejects field corruption, the
coordinator preserves quota/one-shot behavior, and independent paired-array
checking rejects numerical disagreement even when the file hashes agree.

Prepared storage scope: 32 MiB new node-local, 1 MiB shared, zero new shared
files, after prior 518 GiB+144 MiB node and 125 MiB/370-file shared liabilities.
The actual fresh CPU preflight must retain 20 GiB and 5,000 shared files. No
Dataset/model copy, label modification or new training job is involved.

**No GPU execution or competition entry at source freeze.** Read subsequent
original `actual-*.json` receipts before acting. Dispatch through `run_once.py`
with the exact source commit; phases are prepare, push and read-only status.

## Actual dispatch, 19:01 UTC

CPU1002505 prepared the exact 307,111-byte source
`5cc94175a2f2f3c3d5d0814201dda7fa154dbdc01e861c95bd999254a537571e`
from frozen source `f8d2380365e98666aa0d3afb3c3b754b8ebdeb10`. Fresh funding
retained 20 GiB and 5,000 shared files after all declared liabilities.
CPU1002514 then pushed **140075914 v1 once**, private/offline/two T4, slug
`godlydonuts/rsna-context96-fourcore-5cc94175a2f2`. Initial state RUNNING.
No competition entry or Dataset creation. Original preparation/push receipts
are `actual-prepare.json` and `actual-push.json`. Do not repeat either action.

## Actual completed comparison, 19:08 UTC

Original140075914v1 **COMPLETED** in136.09seconds. BothT4s used about1.45GB
peak allocated GPU memory; host peak4.78GB. All non-image context fields were
byte-identical. Maximum image difference1.1921e-7, trained FP32 difference
1.7881e-7, trained FP16 difference2.41995e-5; all fixed tolerances passed.
Four private paired prediction files were independently downloaded/hash checked
and decoded on the owning CPU node, with every numerical difference replayed.

The initial observer1002541 failed before reading these files because NumPy is
absent from its lightweight management environment. Distinct status-only
reconciliation1002565 used standard-library bounded NPY decoding, source4fa4fb4e.
No GPU or notebook rerun, dependency installation, model or scientific edit.

Candidate-first public timing projects to **21,642seconds/6.01hours** for
1,300studies/two lanes, including600setup/1,800tail seconds. The maximum-study
scenario is37,980seconds/10.55hours. These are three public examples, not hidden
runtime acceptance or a controlled speed ratio. Old mean planning was9.44hours.
The result makes this new candidate feasible under the mean scenario; it does
not demonstrate an accuracy improvement or establish the old hash discrepancy's
cause. Original140068692 remains held.

At19:07UTC the account had2,423.82free GPU-VMseconds, four entries today and one
slot. Preserve the original Colab1,920seconds/last-slot allocation. No new dense
entry is made. Read `actual-complete.json` and `agent-decision.json`.

## Conditional exact-entry coordinator

`entry_once.py` provides explicit `check`, `submit` and `outcome` phases for
**existing140075914v1 only**. It cannot create a Dataset/notebook or run another
public GPU session. The original same-weight24-context submission57491555 must
have an actual scored result before this comparison uses a slot. It does not
require the original score to exceed a threshold: the purpose is to measure
this one coverage hypothesis, not tune a series of caps.

The declared entry window isOctober22 00:05–12:00UTC, subject to fresh
authoritative allowance. At least two available slots are required, retaining
one for Colab or a final candidate. It verifies the exact accepted proof, source,
private/offline notebook version, original Datasetversions and actual public
receipt. The cold-mean plan6.01hours and max-study10.55hours are retained in the
entry record; the risk is explicit, not silently called a nine-hour guarantee.

The canonical `entries` record uses an explicit inference-operation key, while
`checkpoint_sha256` remains the actual e606d974 model hash. This permits a
scientifically different inference configuration without replacing the existing
24-context checkpoint entry or breaking account-wide local quota counting.
Only this coordinator owns its record and outcome reconciliation. Durable local
and canonical intents precede exactly one `CreateCodeSubmission` call pointing
to version1's `submission.csv`; no public CSV upload. Uncertain responses retain
the claim and prohibit another mutation.

Nine focused lifecycle tests passed, including pending/failed/unscored original
outcomes, early/late windows, preservation of the last slot and original model
record, one request and an ambiguous-response no-retry case. These are software
checks, not an additional model execution or an accuracy result. Source freeze
and actual nonmutating preflight must precede any later explicit submission.
