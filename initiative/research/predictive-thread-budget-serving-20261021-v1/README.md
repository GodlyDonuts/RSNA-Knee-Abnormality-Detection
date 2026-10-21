# Four-core serving contingency

Prepared after actual CPU140055931v1 showed19.95% less warm preparation time,
with image differences at most1.1920928955078125e-7 and exact remaining fields.
No training, controller, notebook, Dataset or competition entry is changed.

This optional compiler layer retains the full model and current resilient
decoding. Two concurrent study lanes each receive two Torch intra-op threads.
It changes the process-wide setting only between joined preparation pools.

Its own trained qualification must prepare all three original public studies
first at one thread, then at two. All non-image context bytes must match;
images must be finite FP32 and differ by at most1e-6. The original one-thread
contexts remain available for the trained FP32 reference. New two-thread FP32
predictions must differ by at most1e-5, while the existing FP16-versus-original
FP32 and cross-device tolerances also remain. Existing algorithm-equivalence
checks compare original and optimized assembly at the same thread setting;
the separate thread-reference fields truthfully report cross-setting rounding.

This source is a contingency, not a qualified deployment or clinical result.
Actual timing with concurrent GPU inference remains unmeasured. Use only if
the current final runtime admission fails or spare measured GPU capacity makes
the additional actual qualification worthwhile. Current consumers3332/3333
and healthy trainers remain unchanged. Do not silently replace a pushed source,
reuse another qualification, or spend the existing candidates' GPU reservation.

Two focused local tests pass: real context comparison rejects non-image changes,
excess image error and nonfinite images; the actual staged final compiler rejects
missing/mismatched trained evidence and binds the one-thread FP32 reference to
the distinct two-thread candidate. Ruff passes. This is engineering evidence,
not an executed GPU qualification. [Exact preparation decision](engineering-decision.json).
