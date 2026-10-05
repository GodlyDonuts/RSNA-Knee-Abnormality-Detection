# Actual independent native metadata probe

Original873865 FAILED1:0/30seconds on allocated evc22, oneH100/oneCPU.
Actual CUDA_VISIBLE_DEVICES and SLURM_JOB_GPUS both0 were recorded. The NVML
nvidia-smi metadata query timed out after15seconds BEFORE the libcuda driver
version/cuInit calls ran. This is not a driver-initialization acceptance or a
specific root-cause diagnosis. No CUDA context, model, MRI or optimizer ran.
Owning-node CPU873877 collected exact original output; node free~798GiB.
Preserve the original; never repeat the same dispatch or bypass GPU isolation.
No pixels/weights/reports or model results were collected. Source remains pinned.
