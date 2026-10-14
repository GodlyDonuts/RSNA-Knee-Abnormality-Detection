# Actual evc26 admission and device failure

Relocated CPU admission 917440 completed 0:0 and reproduced the original MRI/input/model/RNG/gradient evidence. This is engineering acceptance, not accuracy. Frozen bf46b70e numerical source and accepted labels/masks stayed unchanged.

Original recovery 917001 was cancelled only while PENDING with zero runtime and no allocated TRES. Distinct replacement 917477 failed 1:0 after two seconds before GPU-start receipt, model run directory or optimizer execution. The checked recovery-wrapper SHA matched. The combined preflight assertion initially did not identify which device check failed.

Distinct device-only probe 917508 completed normally as a diagnostic. Slurm assigned GPU 0 and CUDA_VISIBLE_DEVICES=0, but torch reported zero devices, CUDA unavailable and BF16 unsupported; nvidia-smi returned 6 with “No devices were found”. PyTorch also warned of a CUDA initialization unknown error. No device-isolation override, model, optimizer or automatic retry occurred. These observations establish allocated GPU invisibility in this endpoint; the underlying driver/cgroup/hardware cause is not established. Do not treat the diagnostic process exit 0 as GPU acceptance or as a clinical result.

The healthy all-data 908454 on evc40 remains untouched: live check observed 3785 finite updates, epoch 9 underway, no epoch-009 snapshot yet. Exact private evc26 runtime/cache/source copies remain preserved; do not repeat them. A new recovery requires working allocated-device evidence or a separately admitted placement. No production query model was dispatched.
