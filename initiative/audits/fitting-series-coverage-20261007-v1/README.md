# Measured fitting-only series coverage against public anatomical slots

The owner asked to learn from strong public pipelines. Their captured baseline reserves distinct plane/contrast slots, whereas our current frozen selector first ensures plane coverage and then fills six entries by fluid/fat-suppression priority and series UID. This original CPU metadata audit tests whether those policies differ in **available input coverage**, not whether either achieves better accuracy.

The exact original2247 fitting roster was independently recreated and matched the active trainer's UID digest `e9d38ee2…`; no development/confirmation/gold label rows or clinical target values were decoded. Only fitting masks and acquisition metadata were read. The original frozen selector from the active CLS comparison was executed with max_series6/augmentfalse on12476 eligible fitting series. MRI pixels, models and optimizers were not used; nothing in live training changed.

Measured results:

- **214/2247 studies (9.52%)** omit at least one available reference category:103 omit an available sagittal non-fluid series;131 omit an available coronal non-fluid series;20 omit both.
- A follow-up original reader stratified those omissions by the existing reviewed quality tier: **all103 sagittal and131 coronal omissions have an available tier0 series**. They are not explained solely by deliberate tier1 deprioritization. [actual-tier-stratification.json](actual-tier-stratification.json) preserves the identical roster/source bindings and counts.
- **924/2247 (41.12%)** selected bags contain repeated plane/fluid/fat-suppression categories. These are distinct series, not duplicated files; repeated-category views can still be useful.
- All available sagittal/coronal/axial fluid-sensitive fat-suppressed categories are retained under the current metadata. The audit does not show missing coverage for those three categories.
- No eligible fitting sagittal series is marked fluid-sensitive **without** fat suppression in the accepted metadata. This does **not** prove the physical acquisitions lack such sequences. Resolve descriptor/metadata semantics before assuming equivalence with the public non-fat-suppressed sagittal slot.
- The current sampling table has no T1-specific field. Sagittal/coronal non-fluid counts are explicitly **proxies**, not an assertion that they are T1 sequences or numerically equivalent public inputs.

[actual-coverage.json](actual-coverage.json) preserves all category denominators, source/selector/roster hashes and selected-type distributions. It reads no protected target values and establishes no causal accuracy gain. The first original observer880035 refused at an incorrect whole-file-versus-parsed-fold digest check after2seconds; it failed before masks/series inspection, not in training. The distinct corrected reader validates both original byte SHA and canonical parsed-fold digest. Original failure is preserved, not retried or hidden.

## Next matched input experiment

A concrete candidate is to reserve one series for each available complementary plane/contrast category before allocating leftover slots, preserving the six-series/16-slice/384-pixel/effective8-study compute budget. Compare it against the unchanged original selector on the same fitting/development boundary and fixed endpoints. First resolve the non-fat-suppressed/T1 metadata mapping against original acquisition descriptors, then require actual pixel bindings and trainer recovery. Do not change either healthy live job to test this hypothesis. If it does not improve held-out performance, retain the simpler original selection and record that negative.

This mechanism is motivated by measured omissions and the public recipe, rather than by increasing model parameters. It remains a proposed experiment; no new selector was integrated or new GPU training dispatched. Original probe/receipt stay under `/tmp/rsna-fitting-series-coverage-20261007-v2` on evc31; only sanitized aggregates/code are published. A16MiB node-local small-file cap, existing53GiB future liabilities and20GiB free floor were retained; shared artifact growth is zero.
