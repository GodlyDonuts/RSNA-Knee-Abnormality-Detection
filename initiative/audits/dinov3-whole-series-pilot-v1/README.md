# Whole-series GPU pilot dispatch

September 30, 2026. Source `615d0fe4602798a9fe46278b2e5443bfbb49669e`
implements full cached series/slice coverage and neighboring feature context.
The unchanged accepted masks/folds and official DINOv3-L base remain pinned.
All 1,106 source tests and mypy passed before source integration. Model inputs
remain 224 pixels from the existing 256 cache; native detail is not restored.

Bounded Slurm **865318** was submitted once and is RUNNING on evc32. It requests
one H100, two CPUs and 45 minutes. Allocated driver and actual CUDA tensor checks
passed, then full environment/cache preflight and three finite real optimizer
updates passed. Interrupted/resumed recovery and largest-input profiling remain
outstanding. The distinct control is
`/home/sa305415/rsna-knee/whole-series-pilot-615d0fe-v1`, and the clean shared source
is `/home/sa305415/rsna-knee/code-whole-series-615d0fe`.

The harness trains a 16-study/two-epoch diagnostic, independently stops/resumes
another identical run, checks exact model/optimizer/RNG and prediction equality,
verifies all 120 residual projections learned and the 415 original base tensors
are unchanged, then profiles the five largest fold-0 training workloads. Full
recovery checkpoints stay on owned node-local storage. Only a separately typed
compact diagnostic delta below 180 MiB may be preserved on shared storage. This
diagnostic is not competition-eligible, and its tiny evaluation is not an accuracy
comparison. Matched full-coverage control and context production runs remain
prepared only; no new context production job or Kaggle submission exists.

The first local dispatch wrapper failed while parsing its remote Python command,
before the remote intent or `sbatch`. Remote file inspection and corrected Slurm
time-window inspection preserve that evidence; the distinct wrapper correction
returned the one job ID 865318. Preserve both intents and never repeat dispatch.
The initial accounting query used UTC against Slurm's local clock and failed;
the subsequent correct local-time query is in `first-job-check.txt`.

All-development job 865263 continues independently on evc31. Its completed
epoch 1 was privately published and all four objects re-downloaded/size/SHA
verified at HF `8b3bf285d846f93e776f9dc8fa5d877ffd40c861`. CPU retention 865316
verified that redundant local epoch file against the private receipt and removed
only that file, retaining its sidecar and the active recovery checkpoint. The
private epoch remains restorable. The storage gate includes future remaining
primary checkpoints, an atomic checkpoint-write transient, final Kaggle bundle
and ZIP, 100 MiB overhead and the bounded compact pilot; projected reserve is
20.358 GiB and 6,825 files. Another full production job needs a fresh storage
plan; this pilot does not authorize exceeding that budget.
