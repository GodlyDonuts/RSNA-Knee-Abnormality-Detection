# Distinct full224 gradient header recovery

Original866846 FAILED1:0; actual Triton CUDA helper compilation could not find
Python.h. The full224 launcher omitted the accepted header guard/CPATH. Its
original failure/CPU input/source/runtime stay immutable in the old control
and sampling-fixed-final-v1/gradient-failure. No completed gradient measurement
or optimizer update was produced. This is an observed compiler prerequisite
failure, not a model-accuracy failure.

Prepared gpu.sbatch.txt adds the exact215-header/CPUcompile/ABI guard already
used by successful sampling866716, a scheduler-visible CUDA preflight and a
node-local Triton cache cleaned by the owning shell. Numerical4acd6fa source,
three pilot configs,16 training studies, CPU16 tensor receipt and official
weights/runtime are byte-identical. The distinct control is relation-full224-
gradient-header-v2; copy only six original small immutable source/input proof
files, bounded16MiB/48 files, zero checkpoints. Actual phase independently
materializes and rechecks all original16 inputs before loading DINOv3. No
CPU32/newsource manifest replaces the oldCPU16/48-file binding.

GPU reservation requires fresh source/asset stats/oldFailed1:0/error/oldCPU16
and admitted header CPUcompile proof, <=1 other GPU role and funded20GiB/5000
shared reserves after native/sampling liabilities. Stage exclusive intents
and exact script SHA. This preparation is NOT a dispatch or acceptance.
A new actual gradient receipt and training-only scale review still gate the
unchanged production-trainer recovery. Never resubmit the failed old job or
count tiny engineering metrics as competitive evidence.
