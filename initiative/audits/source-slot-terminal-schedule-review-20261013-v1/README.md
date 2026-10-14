# Actual source-slot recovery, corrected terminal schedule review

The original GPU worker completed full/stop1/resume and its complete decoded
model/base/AdamW/LR/counters/all-RNG/scaler/history/CSV comparison, then failed
in the final acceptance checker because it required strictly positive LR. Actual
four-update cosine rates are exactly zero at step4 by the canonical scheduler;
all AdamW counters were4 and scaler growth tracker4. The model did not fail.
Original source, checkpoints, logs and worker return1 remain preserved.

The checker now requires the exact two scheduled rates, allowing zero only when
the declared scheduler returns it;35 focused checks passed. A separate genuine
one-CPU zero-GPU terminal review independently rechecked complete final full and
resumed envelopes, all numerical updates and identical16x12 MRI CSV bytes, plus
the corrected exact-LR condition. No GPU/model training was rerun. The original
GPU comparison had checked the actual partial checkpoint before it was replaced;
the CPU review explicitly does not claim to re-read that overwritten checkpoint.

The first CPU review receipt mistakenly placed the decoded-state object in its
complete_state_sha256 field. Its original immutable receipt remains preserved;
a separate normalized receipt supplies the canonical digest and links the exact
original receipt SHA. No learned state, predictions or acceptance predicate
changed during normalization. The production coordinator binds the original
accepted receipt and this audit retains the normalized hash proof separately.

This establishes engineering recovery acceptance, not accuracy or deployment.
