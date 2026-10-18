# Combined encoder noise and optimizer cost measurement

This distinct prepared wrapper keeps the trained-E3 four-study gradient experiment
and adds a bounded cost comparison of the unchanged ordinary and corrected
training engines. It does not restart the original recovery or alter healthy
training. The existing v1 owner remains unchanged until an exact pre-push
ownership reconciliation permits replacing an unexecuted experiment.

After the noise workers exit, four fresh two-T4 process groups run six updates
each on the original 16 fitting engineering studies, from the exact original
parent: ordinary, controlled, controlled, ordinary. Every rank's initial learned
state and all RNG identities must match across all four runs. The ordinary arm
uses the original readout class; the controlled arm requires the explicit paired
readout. The full noise test still uses the preserved trained E3, not these
temporary parent-start timing fits. Neither phase estimates clinical accuracy.

All step intervals, first-step-excluded ratios and both-rank full trainer wall
times are reported. The step scope includes LR setup, case lookup, image inputs,
forward/backward, clipping, AdamW/scaler updates and loss collectives; it excludes
order construction, progress-file and epoch/checkpoint IO. Full trainer wall time
includes setup and checkpoint IO. No total production-runtime or proven gradient
bias claim follows from a small numerical comparison.

New diagnostic checkpoints remain under a fresh VM-local `/tmp` root until
teardown and are not uploaded, published or used for inference. Admission needs
12 GiB temporary free space for a strict 8 GiB temporary-checkpoint bound; the
working-directory receipt/source/log bound stays 16 MiB. The combined execution
deadline is 2,550 seconds inside a separately funded 2,700-second GPU session.
Each uncertain/failed execution remains original and is never automatically
retried. All original input datasets and weight versions are reused.

Eight new focused checks pass, including complete CPU optimizer-state equality
with and without timing hooks, restoration after failure, shared pre-fit output
admission across ranks, fresh-process and reversed-order source compilation,
temporary-storage policy and refusal to rewrite numerical module bytes. Scoped
Ruff and two-module mypy pass. These are not GPU evidence. The revised wrapper
has not been submitted to Kaggle; no new production run or accuracy is accepted.
