# Fixed paired late-checkpoint average

Preregistered October 18 before the paired final competition score: construct exactly the equal-weight average of completed paired epochs 2 and 3. Both use the same full 3,887-study recipe and original initialization. Do not select weights or epochs using leaderboard feedback. Frozen BN buffers and nonfloating state must be identical; average other floating state in FP64 and cast once. Originals remain unchanged.

This is a cheap empirical test of a flatter inference solution, motivated by [model soups](https://proceedings.mlr.press/v162/wortsman22a.html) and [weight averaging](https://arxiv.org/abs/1803.05407), not an exact implementation of cyclic-learning-rate SWA or a claimed novel algorithm. It can degrade predictions. It retains one model and adds no ensemble inference passes.

The one-shot CPU job constructs a distinct inference-only artifact, then privately publishes and fully re-downloads/hashes all four objects under the canonical lock. It carries no optimizer/resume binding. Separate real MRI, exact T4/offline runtime acceptance and competition scoring are still required. Arithmetic tests do not establish accuracy.

`run_once.py.txt` requires committed source, creates one local intent and never automatically retries an uncertain mutation. Temporary copies use separately reserved RAM; original retained assets are untouched.

Actual execution: Slurm962756 completed0:0 in39seconds. Both original checkpoints were loaded and the derived learned graph was constructed, finite, serialized/reloaded exactly, and privately preserved at revision `7eb8d924ee5e6ec3632608c2d0952dd64c0ec2ca`. All four objects were fully downloaded and size/SHA verified. Model SHA `d8dbf461986975f0300d797a639ef209ac72dafcb7a504d8e088165a75073a61` (292,797,211bytes); 73,089,723 floating source values differ. BN/nonfloating buffers remained exact. New temporary RAM copies were released only after verification; original checkpoints unchanged. Actual MRI/T4 acceptance and accuracy are still unmeasured.
