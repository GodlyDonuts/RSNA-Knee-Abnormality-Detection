# Persistent original-model inference and two-device sharding

The three actual hidden runtime failures require deployment work. The new
standalone ResidentInference loads and validates one exact trained checkpoint/
encoder/model per worker, then reuses that model for subsequent prepared MRI
chunks. Every request preserves source/base/admission signatures, original
samplers, one-study batches, FP32 and the original sigmoid prediction function.
No rsna numerical module or original checkpoint/data/mask/fold was changed.

ResidentPool owns at most two persistent finite workers. It partitions the exact
chunk roster across devices, checks complete shard identities/provenance, restores
original row order and produces the original13-column CSV/metrics contract.
Each worker uses one CPU thread, no nested DataLoader workers. Parent preparation
remains bounded by the existing chunk cache. An overall deadline terminates an
uncompleted worker exchange without pretending a partial output is complete.
No automatic worker restart or CUDA/data-error neutralization is introduced.

Actual synthetic CPU integrations show identical probabilities and roster order
against original infer.run and load each model only once across repeated requests.
A real two-process CPU integration exercises both workers and reuse across two
chunks. This is software execution equivalence, not an actual2T4 speed/RAM pass,
clinical gain, full hidden-scale deadline acceptance or a submitted notebook.

FP32 is fixed; BF16/FP16, larger batches and threaded data workers are rejected
until separately tested. Per-worker host peakRSS, GPU peak allocation and startup
cost are recorded. The notebook bootstrap/package integration, actual hardware/
process-tree memory caps, representative worst-case data and full pipeline timing
are still required before a new competition slot. The9h limit and8h engineering
target are in kaggle-runtime-budget-20261005-v1/requirements.json.

No Kaggle Dataset create, notebook push, competition submission, large transfer
or Slurm/GPU model retry is performed by this implementation. Original failed
GPU/timeout receipts remain preserved. The shared/private publication obligations
and protected gold/confirmation remain unchanged. Do not infer competition
acceptance from tiny fixtures or this prepared API.

Validation:42 focused inference tests and2030 full tests passed; Ruff317,
mypy48, archive and current89/89 evidence checks passed. Complete counters and
remaining hardware/package boundaries are in local-validation.json.
