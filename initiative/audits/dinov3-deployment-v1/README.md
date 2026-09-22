# DINOv3 deployment evidence

The exact runtime at Git `06816e6b698901bd5148d47a3cdf73252c5a3a89` passed GitHub CI and was published privately to Hugging Face at `b3223f263bc5d0b017b18bcc6c83964ba82b9c28`. The overlay publication contains exact Git blobs plus a full target manifest, and declares its accepted checkpoint dependencies. The publisher downloaded and compared all three overlay objects; this does not claim a new full base restoration.

Newton source jobs 838487 and 838496 produced a clean detached checkout. The root independently compared all 69 recorded source/configuration SHA256 values to the committed Git blobs. The pinned Python 3.12 / Torch 2.14 CUDA 13 environment passed fresh offline installation, relocation, CPU forward/backward, tiny random DINOv3 forward and decoder checks in CPU jobs 838473 and 838493. These CPU checks do not certify either GPU runtime.

The initial two-H100 pilot, 838502 on evc38, stopped at preflight after 41 seconds. Slurm allocated two GPUs, the driver inventory exposed one, and Torch could not initialize CUDA. No training or optimizer step occurred. Its script, inputs, dispatch, output, errors and accounting are preserved. Later probes/retries must have distinct receipts and must not overwrite this failure.

The independently reviewed Kaggle successor shares one byte-bound private wheel archive across epoch checkpoints. Its local review passed 76 tests, including 27 independent cases. These local tests are distinct from the actual runtime probes described below; no score or successful production inference is claimed here.

`root-verification.json` maps each original artifact to its preserved bytes. Historical Python files have a `.txt` suffix so formatting checks cannot rewrite original evidence. Original receipts retain their original paths. Large environments and archives remain on private cluster/Hugging Face storage with their exact checksums.

A subsequent bounded probe, 838579 on evc37, passed tiny CUDA tensor allocation on both H100s using the unchanged environment, both directly in the batch and through a Slurm step. Original and PCI bus device ordering both worked with the same scheduler-provided visible devices. This supports a node-specific issue at evc38; it does not prove its underlying cause. The full recovery pilot has a separately recorded retry, and production jobs should exclude evc38 until verified healthy.

The dedicated private Kaggle wheelhouse Dataset and offline T4 kernel both remain at version 1. That actual probe passed fresh installation, CUDA tensor backward, tiny random DINOv3 forward and codec checks. The production shared-runtime guard was then replayed against those actual resources. This certifies the bounded runtime probe, not trained-model accuracy or hidden-test throughput.

The successor packaging code passed all 890 repository tests, Ruff and mypy; its 88 Python source inputs are bound in the root check receipt. Training runtime sources remain unchanged from `06816e6`.

The retry, 838580 on evc37, passed GPU/NCCL preflight but stopped before any training update after 3m08s. The installed architecture expects `model.layer.N.*`, whereas the official pinned checkpoint uses `layer.N.*`. Strict loading rejected the mismatch. Both original weights and complete failure output are preserved. An explicit compatible loader and actual full pretrained CPU check are required before another GPU attempt; this snapshot does not claim the fix.
