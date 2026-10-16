# Exact-value selected-window materialization profile

CPU **934424 completed** on the verified evc39 endpoint. On 16 original fitting
MRI studies, constructing all windows and then selecting 12 took a summed median
**2.1092 seconds**, versus **0.9034 seconds** constructing only those same 12.
Every selected value is bit-equal and global CPU RNG is unchanged. Materialized
window storage falls from roughly 99–149 MB/study to 21.23 MB for these cases.
[Actual profile](actual-CPU-profile.json), [original probe](original-CPU-probe.py.txt).

This is a 2.33x reduction in that substep, not training speed or accuracy. Loading
alone took another 5.3107 seconds across the 16 cases; avoiding window copies does
not remove source decoding/normalization or GPU work. No model/optimizer/GPU ran,
no live trainer changed and no patient records/pixels were exported. This uses
already-funded small node controls and zero shared growth.

A future fast path must preserve the exact original selected-index order, sampler
and all RNG states and pass real full/stop/resume optimizer equality before a
new fit. Never modify the healthy frozen neighbor or queued scientific sources.
