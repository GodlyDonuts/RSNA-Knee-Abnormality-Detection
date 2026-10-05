# Actual batch-versus-task CUDA probe

Original873729 COMPLETED0:0/5:15 on evc48 with oneH100/oneCPU. Both the batch
and explicitly assigned Slurm step Torch subprocesses exceeded the bounded
150-second limit. No final boolean/device/context result was produced, so neither
context is accepted and a specific failure predicate remains unproven. No MRI,
encoder, model or optimizer ran. Preserve exact original source and terminal;
never repeat this dispatch or call it a successful model recovery.
