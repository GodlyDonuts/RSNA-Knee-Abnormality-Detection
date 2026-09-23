# Opt-in MRI series recovery candidate, September 23

The five DINOv3 code submissions failed in Kaggle's hidden rerun with zero
output and no score. Kaggle supplied no hidden traceback. This candidate
addresses one **possible**, unproven failure path; it is not a recovered score.

The completed checkpoint selects at most six series per study. The first
deployment wrapper decoded every series in `test_series.csv` before selecting
six. A read-only count of the archived official `train_series.csv` finds
**734 of 4,407** training studies with more than six series, accounting for
**1,581** series beyond the six-slot model capacity. All three public example
studies have only five series, so the public notebook never exercised this
excess-series path. This count does not assert the hidden distribution or cause
of its failures.

The opt-in `--resilient-series` path in `scripts/kaggle/chunked_infer.py` reads
the checkpoint's bound `max_series` and cache resolution. It pre-caches only
selected MRI series, maintaining the trained plane-first selection on intact
inputs. If a selected series has no DICOMs or raises a recognized DICOM/geometry
error, it tries an alternate series of the same plane before filling the
remaining slots. It passes only successfully cached series to the **unchanged**
checkpoint-bound `rsna.competitive.infer` code. Disk, cache-integrity,
unexpected decoder/environment and model errors still fail. A study with no
decodable MRI series also fails; no text, invented image or clinical label
fallback is used. The original wrapper behavior remains the default unless the
flag is explicitly set in a **new** notebook package.

`tests/test_kaggle_chunked_inference.py` checks intact six-series selection,
same-plane recovery, propagation of cache-integrity failure, checkpoint
config matching, wrapper option flow, and a real synthetic-DICOM failure
followed by successful `CompetitiveStudyDataset` cache loading. Its 13 tests
passed on September 23. This is local functional evidence, not a hidden-scale
runtime acceptance or a valid Kaggle score. The existing corrected epoch-8
private notebook version 1 is unchanged and **held**. Before a submission,
build a distinct immutable private notebook/Dataset from this candidate, run
the exact offline public version, rehearse a large MRI-only roster within
Kaggle's nine-hour limit, verify hashes and the canonical ledger, and use one
live slot only after the remaining failure risks are reviewed.
