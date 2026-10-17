# Preserve the actual whole94 worker during continuation

First full epoch actually completed 486 updates on both T4s and was privately
verified at HFd436986137570a9a4259a4b59981d261cffd2f20/all5objects. Numerical
worker32ef4bbee4f68c07c073e57238523ca40c9ca3618df514a30e454bc91060110c remains
immutable. Original epoch2 preparation stopped before any directory or push intent.

Actual owning-node comparison proves the only reconstructed worker difference is
key ordering inside the literal `_PRODUCTION_PROOFS` assignment after canonical
JSON reload. Every proof value and complete binding is equal; replacing that one
assignment makes all remaining worker bytes identical. This is serialization,
not a changed data/model/optimizer/recipe or a failed fit.

The compiler can now accept the exact previous worker, SHA-bound to its actual
receipt. It permits only canonically equal literal proofs and requires every
other byte equal, then returns the **original bytes**, not normalized new code.
Numerical changes, comments elsewhere, proof-value or bool/int differences,
wrong hashes and use during initialization are refused. Existing compiler cases
and the new regressions passed27tests together with sampled-production tests.

The recovery coordinator is prepared to retain original source modules/control,
canonical ledger, remaining fixed3 schedule and checkpoint/RNG continuation.
Actual CPU replay admission and original epoch2 dispatch must be recorded before
calling it resumed. No model/optimizer checkpoint or dataset is reacquired.
The serving observer's independent filename collision happened after the first
qualification push. Read-only ledger/source/status reconciliation proved exactly
one139136981v1 was already RUNNING; never push it again.
