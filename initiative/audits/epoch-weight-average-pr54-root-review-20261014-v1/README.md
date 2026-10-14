# Exact epoch-weight averaging root review

PR54 f422252c implements uniform effective-weight averaging of epochs6–10,
with epoch10 identity as the comparison. Root integrated only the new module,
tests and README, preserving existing cache/window-bag/live scientific sources.

Root corrected eager DenseDeltaLinear import: immutable older a990 R32 source
trees predate that class, so import is now conditional on the dense method.
Root also made partial-candidate and receipt writes exclusive, preserving any
original interrupted output. Both regressions have explicit tests; focused16
passed. No real averaged candidate, GPU acceptance or accuracy result exists yet.

The intended first source is the completed original public arm of879172:
2247 fitting/1156 outer, fixed10 epochs, internal final0.8997840648. Epochs6–10
are predeclared, and its own epoch10 is the comparator. No per-epoch selection,
gold or confirmation tuning. New candidate rank160 remains one backbone forward,
not a prediction ensemble; runtime and held-out effects remain unproven.
