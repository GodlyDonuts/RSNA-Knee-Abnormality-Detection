# Nine-hour Kaggle deployment correction

The owner supplied the hard per-submission envelope: GPU/CPU runtime each at
most9hours; two16GiB T4s, four CPU cores/~29GiB hostRAM alongsideGPU, ~30GiB
CPU-onlyRAM; internet disabled; exact submission.csv output. requirements.json
records it. The proposed engineering target is at most8hours end-to-end to leave
headroom. This target is not an achieved benchmark or guaranteed hidden result.

Authenticated live outcomes and the single Newton canonical ledger now agree:
56833260 native all-development epoch10 scored0.802;56833608 seed1 reviewed-label
candidate scored0.685.56833454 seed1 control,56833935 seed2 control and56833831
seed2 candidate each failed for exceeded runtime, no score/zero output. Prior
native epoch5 remains0.804. The missing partners prevent a complete paired Kaggle
supervision comparison. Do not call timeouts pending or treat missing scores as0.
No competition/notebook/Dataset mutation was performed; live five slots remain.

The exact old runner uses FP32 on the default single CUDA device, serial DICOM
preparation and a fresh model/encoder/checkpoint/source load for each32-study
chunk. Healthy slices are decoded once by healthy_slices and again by cache_series.
These are source-proven inefficiencies, not a measured breakdown of the hidden
timeout. Two identical-architecture jobs did finish, so execution variability
and margin matter. Three public cases cannot admit the hidden-scale deadline.

Public pipeline times: native98.43s, seed1control101.54s, candidate98.09s for3
studies, of which53.18/56.55/54.40s belong to model subprocesses. These include
startup and heterogeneous tiny cases; do not extrapolate them as a full-cohort
benchmark. They do not include every notebook bootstrap/dependency-install cost.

The first optional code change, --single-decode in chunked_infer.py, tries the
normal decoder once using frozen original bytes and invokes the old slice-level
recovery only if it fails. It must preserve exact retained MRI arrays, metadata,
geometry, bad-slice recovery and fatal infrastructure handling. Existing source
notebooks/checkpoints are immutable and have not been overwritten or resubmitted.
This optimization alone does not claim to fit nine hours.

Next deployment work is resident model loading once per device, two T4 workers
and bounded CPU/RAM/cache pipelines, preserving actual sigmoid probabilities.
Benchmark representative large/long studies with total startup/decode/inference/
CSV time, process-tree RAM, both GPU peaks and throughput. No new competition
slot should be spent as a runtime diagnostic; use a private offline performance
notebook. Any precision optimization is a separate numerical acceptance, never
an untested BF16/FP16 substitution. Actual T4/full-scale acceptance is still absent.

Local validation:33 focused inference tests passed; exact healthy and mixed-bad
DICOM arrays/metadata match the legacy path. Healthy decoding calls fell from6
to3 in the three-slice fixture. Full2021 tests/Ruff312/mypy48/archive/current89/89
passed. This demonstrates engineering equivalence/cost reduction, not full T4
speed, memory acceptance or clinical improvement.
