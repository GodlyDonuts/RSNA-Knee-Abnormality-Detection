# Candidate vectorization of the new path serving score

New inference work while original955582/955711/955712 wait for allocation. No
queued trainer, recipe, checkpoint, mathematical source module or dataset was
changed. No GPU reservation, notebook or competition submission was made.
This is a speed mechanism, not a claim of new diagnostic accuracy.

The original start-conditioned marginal null computes every possible start's
prior with separate matrix-vector calls, twice per witness score for actual and
zero evidence. It also loops over starts separately in its backward recurrence.
The candidate advances all active starts in one batched matrix multiplication
per source step, computes only valid triangular emissions in bounded chunks,
and batches start states during each backward step. Fixed metadata work is
reused for actual and baseline scores. The physical arm retains its original
stable log-semiring recurrence; no exp/matmul shortcut risks sparse underflow.

Variable transitions, conditional-on-start marginals, gaps, start/stopping
policy, focal singleton paths and numerical zero subtraction remain. A separate
FastPathReader preserves the nine learned tensors, five evidence summaries and
four geometry features. It does not learn different weights or change slice
sampling. Both original full jobs still execute their pinned reference source.

The earlier serving module has an explicit vectorized_scoring=False default.
True selects this candidate from the explicit scientific bundle import path;
receipt names its execution version. Original exported head bytes are identical.
Do not enable it in a competition notebook without actual MRI/T4numerical and
runtime evidence for that exact bundle and head. No default policy changed.

Thirteen NEW focused synthetic CPU tests pass: FP64score/gradient agreement for
both arms across singleton/varying directed/sparse paths, FP32extreme and zero
evidence, work-bound rejection, complete reader-summary/output agreement and
strict identical-parameter optional-serving selection. Only this new arithmetic
and serving glue were exercised; no old MRI/model recovery or cohort audit ran.
Ruff passes on the new files and changed serving module.

One-thread CPU synthetic timings (144tokens/12targets, three measured repeats):
16windows8.87ms reference versus3.24ms candidate;32windows26.51ms versus9.37ms.
Prior calls reduce240matrix-vector calls to15batched matrix calls, and992to31,
respectively. FP32score differences were0and4.77e-7 in these two synthetic cases.
These approximately2.7–2.8x arithmetic timings do NOT establish whole-MRI speed,
CUDA/T4throughput, VRAM peak, clinical accuracy or a nine-hour deployment pass.
The forward workspace estimate is not an autograd/global CUDA memory guarantee.
