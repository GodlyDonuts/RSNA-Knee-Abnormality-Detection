# Fixed late-average versus E16 on the same raw-MRI outer endpoint

Hypothesis: the predeclared equal learned-parameter average of E12/E14/E16 improves
ranking over unchanged E16 while retaining final E16 BatchNorm/saved buffers.
A second question is how raw Kaggle serving differs from the original cached/BF16
internal endpoint. Compare **both fixed models on identical decoded windows** so
an input change is not mistaken for an averaging gain.

- Same 1,156 weak-label outer studies; exact original fitting/outer boundaries.
- No gold/confirmation selection or clinical labels/reports uploaded to Kaggle.
- Same six-series, uniform16, selected-slot triples, native384 and full aggregation.
- One MRI preparation and window tensor feeds both separate complete models.
- Actual both-T4 FP32/FP16 comparisons against original CPU references precede
  evaluation: maximum probability tolerances .0003/.003 respectively.
- Paired whole-leakage-component uncertainty and all 12 targets after execution;
  report undefined bootstrap draws and rare support. No best-epoch picking or
  averaging-weight tuning. Use the existing fixed component-bootstrap protocol.
- Fixed average SHA `5cc0e5bffb05b3b73b25190eec4fcdb458f07908b570de2091b893a2ae374ac8`;
  E16 SHA `5bec977964231fe67009d7106ad63ec71fdbd9cde5ddeb2cc2883ac59b0a7be5`.

The new loader/runtime are separately scoped; original scheduled inference still
rejects the derived engine. This research runner cannot write submission.csv.
Two persistent T4 workers reuse both models; original four-study bounded prefetch
and cleanup remain. A six-hour notebook budget includes a 20,700-second inference
limit; no hidden-scale nine-hour success is inferred.

CPU 934289 failed before source extraction/copy/model/cloud due an omitted base64
import in its standalone bootstrap. Original empty output is retained. Distinct
corrected **CPU 934300 passed**, binding exact original roster and already fully
verified average. [Receipt](CPU-component-prepared.json). New component/ZIP budget
is 2 GiB; original obligations and reserves remain, no deletion credit.

Private Dataset **12630620 v1** was created once. GetDatasetStatus returned 403;
authenticated GetDataset independently confirmed the exact private ID/version.
The original controller stopped after create; a distinct finite continuation
reused the original creation/source receipts and fully download/size/SHA verified
all ten objects. It pushed the exact private offline two-T4 notebook once:
`godlydonuts/rsna-fixedavg-e16-outer-5cc0e5bffb05` v1, kernel 139017805.
[Push receipt](exact-private-T4-pushed-once.json),
[whole-byte verification](private-Dataset-all-10-objects-verified.json).
No duplicate create, no GPU result, no accuracy gain and no competition submission
have been claimed. Raw predictions will go directly to private cluster storage;
only sanitized receipts/aggregate statistics belong in Git/HF.
