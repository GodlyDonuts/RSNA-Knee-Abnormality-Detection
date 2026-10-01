# Bounded official-weight sampling acceptance, October 1

The owner's standing creative directive is implemented as a falsifiable
observation-coverage experiment: epoch-varying stratified slice bins versus the
same uniform grid, keeping the series and optimization budget fixed. This
pilot establishes engineering acceptance; its16-study/two-epoch metrics do not
measure competitive performance. Native384 full-cohort production remains the
next primary experiment on frozen C9; its live source lanes were not modified.

A separate CPU preparation job **866610** was submitted once, requesting one
CPU,16GB RAM and zero GPUs. Control:
`/home/sa305415/rsna-knee/stratified-pilot-7ac7a97-v1`. It verifies the exact source,
existing official weight/config and runtime/cache archives, accepted release and
fold replay, plus actual decoding of16training and16outer-fold studies. The
existing4.079GB pilot archive is reused; no image or weight download occurs.
Large extracted caches, environment and sparse code live only on owned node-local
disk and are cleaned by their creating invocation. No shared environment/cache
copy is made.

Scientific source is main **7ac7a9779d3c8bfd756a44eceb6414e9bb763b85**, with the
new stratified policy. `source.json` binds45declared code files. The9,232,232-byte
Git source pack contains all current trees/declared source blobs and new objects
since C9; the existing immutable C9 object store supplies historical objects.
Its exact bytes are stored on Newton at the control's `source.pack`, with digest
in `source.json` and `payloads.json`. This is a sparse source projection, not a
complete repository backup. The local projection prototype passed exact declared
membership, byte hashes and clean status. Earlier local incomplete prototypes
were not staged. The archived rebuild helper recreates a new private output;
all authoritative source files remain available through pinned Git history.

The CPU job completed **0:0** on evc21 in4minutes25seconds. Its actual
32-study decoding and full asset/source/release/fold proof is preserved in
`cpu-prerequisites-verified.json`; terminal/stderr in `cpu-terminal-receipt.json`. Git emitted warnings
about historical alternate nesting and one nonmaterialized ConvNeXt config blob;
the declared scientific source bytes passed independently. This observation is
not a GPU or data result. Logs preserve those warnings; this is not a full historical Git object-closure
claim. The complete declared scientific files, CPU terminal0:0 and acceptance
receipt passed before GPU dispatch.

The prepared GPU script requests one H100/two CPUs and at most45minutes. It was submitted once as **866616** after exact CPU acceptance and a fresh
funded reserve review; the initial actual state was **PENDING (Priority)**.
Preparation alone was not treated as permission to bypass acceptance. It checks allocated CUDA/NVML,
Python headers and exact CPU proof before any optimization. Both policies run
four updates; the candidate is additionally interrupted after the first update
in epoch0 and after the first update in epoch1, then must produce the same full
checkpoint/optimizer/RNG state and byte-identical outer predictions as its
uninterrupted run. The verification reloads inference under uniform sampling,
checks all120learned residual modules and415unchanged official tensors, and
compares initial learned tensor hashes plus actual zero-adapter evaluation logits.
These checks cannot demonstrate an accuracy improvement.

Before any later GPU dispatch, read actual Slurm/logs and the CPU terminal
receipt, recheck payload and asset stat identities, at most two GPU nodes, and
fund every outstanding native/source obligation plus this pilot. Additional
funding is145,584,750bytes and50files, including a bounded diagnostic checkpoint;
retain20GiB/5000files. Existing intent requires read-only reconciliation; never
repeat staging or jobs. Competition submissions are not part of this pilot.

Payloads are frozen by `payloads.json`; `run_phase.py.txt`, `cpu_data.py.txt`,
`cpu.sbatch.txt`, `pilot.sbatch.txt` and `verify_initialization.py.txt` are exact
prepared source. The source pack is a pinned external cluster artifact and can
be rebuilt from the archived helper; do not mistake preparation for dispatch.

The local suite passed1,434tests in109.99seconds, including21focused sampling/
initialization cases, with Ruff205files, mypy38modules and archive/v2 evidence
checks. No new MRI accuracy or Kaggle submission is claimed by these checks.

## Actual first GPU failure and distinct node recovery

Job866616 terminated **FAILED2:0** on evc22 in2:35. Its allocated CUDA/NVML
probe exceeded the bounded timeout; the second Slurm-step probe exited124.
Stdout was empty, no header verification or training/checkpoint output existed,
and zero optimizer updates occurred. This is a failed hardware-admission attempt,
not a trained model result; the underlying node failure is not proven. Exact
terminal/stderr and absence evidence are in `evc22-preflight-failure.json`.

After read-only terminal reconciliation, fresh asset/payload checks and funded
quota review, distinct job **866624** was submitted once using
`pilot-evc31-v2.sbatch.txt`. Its only launcher difference is the job name and
fixed evc31 node, where prior deterministic DINOv3 acceptance succeeded. Scientific
code/config/cache/runtime and CPU acceptance remain unchanged. The initial
actual state was **PENDING Priority**. Its launch digest is
6da174a996c0108b39e164f77fbf42a68553f17732877511b512ce25939370af.
Do not retry/restart either job or mistake a submitted script for optimizer work.
