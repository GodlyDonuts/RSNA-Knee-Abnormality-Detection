# Actual 1300-study exact FP32 runtime result

Original private offline two-T4 kernel137329288/v1 COMPLETED. All1300x12
probability values exactly matched the original saved FP32 reference; finite
predictions, zero fallbacks, one model load per device and162 prefetched chunks
were verified against exact source/version/checkpoint/roster identity. No
reference inference was repeated and no new Dataset/competition submission was
created.

Pipeline elapsed time11508.250910209seconds (3.197h), versus original reference
18303.5986seconds (5.084h), about37.1% less time/1.59x throughput. The model and
MRI input/pixel/selection policy remained unchanged. This is genuine runtime
evidence at1300 fitting-only studies, not a new accuracy result or proof about
the unknown hidden-data tail.

Both GPU peaks were2360713216bytes. The old reported host figure8174104576bytes
omits both preparation workers and the resource tracker; it is **not a complete
process-tree upper bound**. Do not use it for final RAM admission. The separately
prepared notebook-inclusive memory check was pushed ONCE after this exact
terminal, fresh quota/job/account/private-source checks:137358409/v1,
godlydonuts/rsna-fp32-prefetch-memory32-85bfc1ce-v2, sourcec8e63bd416240602.
It is currently unscored engineering execution, not a competition submission.
Never repeat the push or upload its fitting/public CSV to the competition.
