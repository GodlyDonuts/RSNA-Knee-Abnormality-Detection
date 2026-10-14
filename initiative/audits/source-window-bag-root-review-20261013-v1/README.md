# Source-window bag root review

PR50 head d448ae966512e02c899f1fa970a1208ee0f47d87 was reviewed and applied as
an incremental trainer dispatch plus new modules/configs/tests. Current live
4fedbec3/a990/1bcd numerical sources remain frozen on their owning nodes.
The source_slot.py bytes are unchanged. Root retained the terminal cosine-LR
checker correction and the independently staged checkpoint publisher.

The optional mechanism trains one study-level loss over the same four windows
in both new arms; one uses fixed mean probability pooling, the other learned
disease-query window weighting. They share parameter graph, initialization,
RNG/input views/supervision/schedule. This changes exposure versus the live
one-window fit, so that fit cannot be reused as a matched mean control.
Training has approximately four times the encoder views; resource/performance
admission and actual GPU recovery are still required before either new fit.

Root added refusal of malformed context rank and nonfinite queries, including
in the mean control. The first focused run saw an archive membership failure
before manifest refresh; it was interrupted when the source changed. That run
is not an accepted test result. Committed-source focused/full checks follow.
No GPU, production fit or accuracy acceptance exists for either bag arm.

Committed-source focused checks passed:169 passed,1 skipped; Ruff482/mypy64/archive and current v2 89/89 complete. Full suite is running; no full-suite/GPU/accuracy acceptance claimed yet.

Full committed suite passed3138 tests,5 skipped. The candidate is implementation-accepted; real MRI/common initialization, GPU recovery, performance and funding remain separate requirements.
