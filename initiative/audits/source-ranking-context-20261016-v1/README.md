# Prospective pairwise-ranking implementation

Separate versioned source; frozen original/context/confidence/full-parent
engines and live controllers are unchanged. Eighteen focused tests passed,
including closed-form binary loss, an independent fractional pair loop, actual
two-view probability ordering, masked NaNs/ties/no-pair fallback, finite extreme
logits, complete update-boundary recovery and uniform original-engine equality.
Ruff and mypy3source checks passed. These are software checks, not actual MRI
acceptance or disease accuracy. The fixed experiment is declared in
[the protocol](../../experiments/competitive-v1/source-ranking-context-v1/README.md).

Actual CPU staging, full32feature engineering, full3,887study fitting and private
epoch publication have not yet occurred. No new GPU or competition submission
is requested by the implementation itself.
