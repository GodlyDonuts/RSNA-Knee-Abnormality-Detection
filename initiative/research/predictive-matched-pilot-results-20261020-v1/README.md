# Original matched-pilot result collection

`collect.py contrast` targets only original GPU984784. `collect.py generative`
targets only original GPU986011. A live job returns its actual scheduler state
without an intent or CPU allocation. A successfully completed job is read once
using one CPU and zero GPUs on evc27. No job is submitted, restarted, cancelled,
or promoted by this collector.

The complete original aggregate acceptance bytes are retained without
reserialization. Their source, binding, scheduler completion and run-start
identity must match the original submission. Weights, case IDs, clinical
labels, individual predictions and MRI arrays are not collected. Any existing
collection intent is reconciled rather than rerun.

The report computes the predeclared first20, peak126–150 and final20 update
windows for both roles. It presents update means and study-weighted means
separately, because the actual pilot batches alternate eight and four studies.
It preserves clipping rates and gradient norms. These are training diagnostics
on twelve fitting studies, not clinical accuracy or automatic promotion.

At22:42UTC both source preparation jobs and contrast984784 are RUNNING;
G100986011 is PENDING Resources. Original emitted cache progress has reached
at least3,300/3,887 studies in both streams. No terminal result has yet been
collected by this script. See [actual live state](actual-live.json).

Five local analysis tests passed; these check aggregation and rejection of
incomplete numerical histories, not a model's performance.
