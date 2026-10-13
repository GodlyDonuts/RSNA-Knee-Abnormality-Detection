# Public DINOv2 input reference (Pilkwang v15) — source-gated, review pending

Original work: [`pilkwang/rsna-knee-baseline-v1`](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)
version 15 by Pilkwang. Nothing here re-implements that notebook; the module only
loads explicitly reviewed top-level nodes from the **exact** retained source.

| Binding | Value |
|---|---|
| Source bytes / SHA256 | 155668 / `b32a9155fcc73c519e75e78c08e07d30393efc591c2798a3e40ca92f39832bb8` |
| Collector cell-join (`extracted.py`) SHA256 | `8fb028d3de3d29b3a6cc2463732013b8efbe290e56b5e0dd0bb7f10cadd758f1` ([source-binding](../../../initiative/audits/public-recipe-gap-review-20261008-v1/source-binding.json); join rule from `collect.py.txt`) |
| Already reviewed model nodes | `SlotHead`, `Model`, `fingerprint` (probe `a772b638…`) |
| Public member | `m_e5427d6c21.pt` `7223cd50…6ab33` (private on evc31; do not re-download) |
| Renderer node review | **pending** — see `review.json` |

## Status

The authoring session could not obtain the exact source: it is not in Git or the
HF mirrors (only the owner's ignored
`artifacts/public-solution-intelligence-20261007/pilkwang--rsna-knee-baseline-v1.source.txt`
and `extracted.py`), and the Kaggle endpoint was not reachable. Per issue #12 no
renderer, selection or windowing code was guessed. `load_reference` raises
`SourceReviewPending` until `review.json` names the reviewed nodes, their unparsed
text digest, and the entrypoint/output-key mapping.

## Completing the review (bounded source handoff)

1. Place the retained source on the reviewing machine; `provenance.verified_code`
   must reproduce both digests above (any mismatch is reported, not tolerated).
2. Read only the renderer, series-selection, slot-assignment, windowing and their
   constants/helpers; list them in `review.json["nodes"]` and set
   `preamble_modules`, `entrypoint`, `entrypoint_call.output_keys`
   (`images`, `slot_mask`, optionally `slot_series`, `slice_order`).
3. `extract_nodes` refuses nodes that import outside numpy/pydicom/torch/cv2/PIL/
   math/os/re, reference `open`/`eval`/`pickle`/`pandas`/`load`, or contain
   label/annotation/gold/URL strings. Top-level cells are never executed.
4. Record `node_text_sha256`, set `status: reviewed`, flip `source_verified` in
   `contract.RECIPE` for each constant the reviewed text binds, and resolve every
   `contract.UNBOUND` item (slot order, missing-slot fill, ordering/laterality,
   intensity, resize, layout, window aggregation) from that text.
5. Add original-vs-contract tests over `synthetic.CASES` (anisotropic spacing,
   both lateralities, oblique, scrambled filenames, absent slots, duplicates,
   constant images, varying lengths) using `compare.compare_renderings`, which
   reports tensor, slot-mask and ordering equality separately. Original fallback
   quirks are parity; improvements belong elsewhere.

## Prepared owning-node comparison (not run)

`scripts/newton/compare_public_dinov2_inputs.py` — evc31 CPU only, refuses
otherwise and while review is pending. Fold-0 fitting population only (2247/1156
boundary re-checked), roster of ≤64 chosen by salted SHA256 of study UIDs (no
outcomes read). Compares public-rendered tensors with current `CompetitiveStudyDataset`
inputs (16 slices, 384 px, 6 series) and exports only `compare.sanitize`d
aggregates: FOV (130 mm vs full header field), slot occupancy/overlap, slice/
channel coverage, finite min/max quantiles, refusal counts by exception type.

## Public member inference route for the next end-to-end test

study DICOM dir → reviewed v15 selection/render entrypoint → `[6 slots, …, 3, 336, 336]`
+ slot mask → reviewed `Model`/`SlotHead` (DINOv2-small backbone, CLS + mean-patch
pooling) with `m_e5427d6c21.pt` loaded `weights_only=True`, strict → per-window
logits → reviewed window aggregation. Dependencies: the exact source, the private
checkpoint, the DINOv2-small base weights the reviewed `Model` constructs, torch
2.14, pydicom, numpy, plus any cv2/PIL resize the review binds.

## Evidence boundary

Synthetic CPU tests here prove gating, extraction safety and fixture geometry only.
They are not real-MRI parity, GPU parity or Kaggle accuracy. Public high-score
wrappers are ensembles with shared ancestry and titles are not submission
receipts; no 0.95 single-model claim is made.
