# Early predictive-model T4 benchmark

This is one bounded private, offline, two-T4 engineering execution on the three
public example MRI studies. It is not a trained model or a competition submission.
The exact original public MRI parent is reused from Dataset 12431568 version 1;
no new model Dataset is needed. Own modules and the privately attributed parent
loader are embedded with exact hashes.

The benchmark decodes complete DICOM studies, constructs the same two masked
contexts and complete descriptor/support rosters, and times FP16 inference on
both T4s. Fresh zero-shift heads only test context-prior precision, so a separate,
explicitly synthetic nonzero-shift check exercises the evidence path and restores
the original state. Neither check qualifies a future trained checkpoint.

`run_once.py.txt` supports prepare, inspect, push and status. The push requires
fresh free allowance of at least 3,600 VM seconds: a 600-second hard probe cap
plus the existing 3,000-second source-zero final-serving reservation. It uses the
canonical Newton account claim/ledger and registers its exact version under a
capacity-scanned family. Existing intent or uncertainty forbids repeating a push.
No paid credits or competition slots are used.

An earlier read-only account inspection failed on an absent remote capacity
module, before mutation. Its original receipt remains in ignored artifacts.
The current controller embeds the actual local capacity implementation instead.

This public workload cannot establish hidden-size runtime or tail-case admission.
The eventual trained candidate still needs exact checkpoint/source parity,
representative and tail-case measurements, and complete offline output validation.

Local validation: 21 DICOM parity tests, three benchmark protocol tests, and 26
missing-source continuation tests pass. The latter helper is not used by this
benchmark and has not dispatched a continuation.
