# Reproduced full-data CUDA stall and bounded data probe

The first fold-0 production run, Slurm job 840562 on evc49, completed 22
optimizer updates and then stopped making progress before its next logged
update. A single checkpoint-stop request reached both workers, but the
120-second grace expired while one rank remained in CUDA synchronization.
The job ended `FAILED 2:0` without a checkpoint or complete epoch. The
[first-attempt audit](../newton-production-stall-v1/) preserves that run.

The same accepted release, official pinned DINOv3-L weights, committed
training source `c58090e7c7a238461a5ac69293b32b64b74c6721` and production
configuration ran once more as job 840578 on **evc40**. Its allocated-device
driver check passed. All 22 non-time optimizer events matched the first run,
including update 22 loss `0.16586126387119293`. It stalled before the next
logged update too. One checkpoint-stop request again reached both workers;
after the grace period, Slurm reported `FAILED 2:0` and no checkpoint existed.
The cross-node repeat rules out a simple evc49-only explanation, but does not
yet identify the CUDA, distributed, or input operation responsible. Neither
attempt completed an epoch or produced a Kaggle score.

`event-comparison.json` contains the exact event comparison. `v6-originals/`
contains the original job header, run record and stop request, plus losslessly
compressed stdout and stderr. `remote-originals-manifest.json` and `files.json`
bind their byte sizes and SHA-256 hashes. The original first-attempt records
remain in its linked audit.

The deterministic fold-0 sampler identifies eight studies and 98 cache files
for zero-based update 22, the next update after the two stalls. `cpu-probe/`
contains the exact request, source-pinned extraction and validation script,
one-CPU Slurm script, stage proof and dispatch receipt for job **840735**. It
used no GPU and extracted all selected files from the previously verified
82.8 GB private cache archive. Its first strict CPU image-range assertion
failed (`FAILED 1:0`); the original traceback and Slurm record are preserved.

The separate one-CPU value probe **840800** completed `0:0`. Its full result
and byte receipt are in `cpu-values/`. Seven sampled studies had finite pixels
inside `[0,1]`. In the rank-1 third microbatch, one study had **20** pixels
at `1.000000238418579`, with no negative or nonfinite pixels. This exceeds
the model's strict normalized-input guard by two FP32 units in the last place.
The bound source cache is uint16, and CPU bilinear antialiasing can round
normalized white pixels above one. This is a concrete input-contract defect
at the repeatable stall boundary, although a causal GPU diagnosis still
requires a bounded two-rank CUDA check. `rsna/competitive/data.py` now clips
only the interpolation result to `[0,1]`; the accepted source data, labels,
masks and folds are unchanged. Do not relaunch the unchanged full-data job.
