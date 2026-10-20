# Predictive cache stopped on an already corrected source defect

At 17:01:46 UTC on October 20, raw job **980960 failed 1:0**. Context job
**980969 failed 1:0 at 17:02:00** after observing its producer's terminal
failure. Cache owner 91707 stopped naturally without retry or cancellation.
Neither job has a complete-cohort receipt. The last progress counters were
1,747 raw and 1,748 contexts; these counters are not a reconciled sealed-prefix
inventory because already-running workers may finish after another fails.

**The source pin was the mistake.** The new pipeline used baseline mirror
revision `52f995f4568e36bfa5fbbaf69a89af41226b4f5d`, despite the existing
authoritative two-file correction in
[`source-pixel-representation-v1`](../../audits/source-pixel-representation-v1/).
No new pixel interpretation or discarded image is needed.

CPU **981798** reproduced the original strict decoder's failure on retained
source bytes. CPU **981799** decoded 187 of the case's 188 DICOMs; the remaining
file exactly matched its original expected size and SHA, but advertised a
640×640, 16-bit frame while containing only 409,600 of the required 819,200
pixel bytes. This is not an interrupted download. CPU **981819** matched the
two known bad files to fitting ordinals 1740 and 1850. All seven source-table
fields match the accepted plan; the inventory additionally has a descriptive
`storage` column. Both affected cases remain unsealed.

CPU **981840** then used the already accepted official bytes in RAM, with
unchanged numerical decoding, normalization, conversion and assembly. All
**188 frames** passed through all three stages, producing three matrix groups.
There were no source HTTP requests, remote writes, label changes, skipped
frames, header edits or synthesized pixels. The original malformed bytes remain
preserved.

- [Original terminal and retained-state inventory](actual-readonly-failure.json)
- [Strict original reproduction](actual-original-reproduction.json)
- [Complete retained-frame diagnosis](actual-original-frame-diagnosis.json)
- [Source-schema comparison](actual-source-schema-comparison.json)
- [Actual authoritative-byte regression](actual-authoritative-overlay-reproduction.json)
- [Read-only reproduction source](probe_official_overlay.py.txt)

The accepted source plan SHA is
`68b44c1412ced183db223220f39cd4a6e071d40515df4c7b75b40a96ba6d0f3d`.
Its original private-HF receipt SHA is
`2c88c4f53cb3d2332e8079e1cd353eede5a601172febf0ab0b74ce8161f9fbb4`;
the authoritative correction revision is
`0b5e410288232f570c69ba4f44723356e6ae6ab7`.
Effective input provenance must explicitly bind the baseline plus these two
replacements; it must not silently relabel old source or cached arrays.

Two read-only observer defects are preserved separately: CPU981818 stopped at
an overstrict row-schema comparison before pixel decoding, and CPU981829
stopped while serializing a `SourceGroup.frames` attribute that is actually
named `raw`. CPU981840 fixes those observer defects. They are not additional
training or data failures.

Next action is a distinct missing-only cache successor that reconciles all four
existing generations, reuses every sealed unaffected case and complete source
member, applies only the exact two authoritative replacements, and binds the
new input lineage into fitting and serving. The full 3,887-study roster, labels,
masks, initialization and B/C/C0 numerical recipe remain unchanged. No full
predictive training or accuracy result has been produced by this recovery.
