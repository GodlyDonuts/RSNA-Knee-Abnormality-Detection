# Measure the report label and teacher mixture signal

This new CPU-only audit uses all3887 already-admitted fitting target rows and their
corresponding two-view parent logits. It reads no MRI, clinical report, held-out
cohort, gold, checkpoint or learned features. No labels, live losses or model
parameters change. There is no new GPU allocation or model transfer.

The production objective mixes25% observed report label with75% teacher
probability; masked labels use the teacher. Count how often teacher and binary
weak label disagree, including both-view and uncalibrated0.9-probability cases,
and how often the mixture target stays on the teacher's side of0.5. Four fixed
source weights are an algebraic sensitivity report, not a recipe sweep, accuracy
metric or a claim that weak labels are right. The mixture's pointwise optimum
is NOT a hard bound on the trained network's predictions or ranking performance.
The exact loss derivative at parent logits is tested against anchored_loss; it
is not an actual head/encoder gradient measurement.

The result can motivate a later matched source-weight experiment; it cannot
choose the winning weight, diagnose label errors or explain the Kaggle gap by
itself. Existing paired and higher-LR comparisons finish unchanged first. A new
64KiB control child fits within the original paired8MiB envelope only after
funding all still-unwritten epoch sources and the1MiB scoring child; fresh node
and shared reserves are checked before any output. All output is aggregate.
