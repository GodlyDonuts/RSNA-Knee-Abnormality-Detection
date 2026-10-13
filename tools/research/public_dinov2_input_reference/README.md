# Public DINOv2 input reference (Pilkwang v15): reviewed source extraction

The original work is [`pilkwang/rsna-knee-baseline-v1`](https://www.kaggle.com/code/pilkwang/rsna-knee-baseline-v1)
version 15, by Pilkwang. We use it for private research only; redistribution licensing
has not been independently confirmed (see the handoff `provenance.json`). Every pixel
decision here is made by the original functions, which are parsed out of the exact
retained notebook. None of them is re-implemented.

| Binding | Value |
|---|---|
| Source handoff | [`initiative/audits/public-dinov2-source-handoff-20261013-v1/source.notebook.json`](../../../initiative/audits/public-dinov2-source-handoff-20261013-v1/) (main `e480af4b`) |
| Source bytes / SHA256 | 155668 / `b32a9155fcc73c519e75e78c08e07d30393efc591c2798a3e40ca92f39832bb8` |
| Collector cell-join SHA256 | `8fb028d3de3d29b3a6cc2463732013b8efbe290e56b5e0dd0bb7f10cadd758f1` (reproduced from the handoff) |
| Reviewed node text SHA256 | see `review.json["node_text_sha256"]` (47 nodes) |
| Public member | `m_e5427d6c21.pt` `7223cd50…6ab33` (private on evc31; do not re-download) |

## What was extracted (`review.json`)

The extracted nodes cover the constants and helpers below, plus `HDR_TAGS`/`probe`,
`annotate`, `pick_slots`, `side_from_geometry`/`side_from_corner_x`/`lat_of`,
`order_slices`/`_order_dominant_axis`, `read_slot`, `normalise_laterality`,
`build_cache`, `window_starts`/`predict_member` and `adopt_config_globals`/`WeightsError`.
The constants and helpers they depend on are extracted with them.

The following are **not** extracted: the report lexicon, `read_labels`/label tables,
annotated/gold handling, training, `plan_cache` (it reads train/test CSVs), `walk`,
weight-package search/`torch.load(weights_only=False)`, submission writing and `main`.

The gate refuses non-allowlisted imports, `open`/`eval`/`pickle`/`pandas` names, loader
attributes and label/annotation/URL strings. Four nodes conflict with it. Each has an
exact, justified exception recorded in `review.json["reviewed_exceptions"]`:

- `annotate` uses `pd.to_numeric` on probed header strings. `pd` is bound to
  `provenance.MetadataPandas`, a metadata-only adapter that exposes `to_numeric` and
  nothing else.
- `build_cache` uses `json` only inside its `ORDER_CACHE` branches. `ORDER_CACHE` is
  bound to `None` (its value on the scored platform), `Path` is a refusing placeholder,
  and the runtime import guard still refuses `json`.
- `WeightsError` and `adopt_config_globals` contain two strings matched by exact SHA256:
  a docstring that names `LabelSourceError`, and a message where "cannot" contains "annot".

`reference.py` supplies only what the notebook's top level provided:

- `header_frame` is the `walk` frame for explicit study directories, with the same
  `probe` and scandir order.
- `configure` picks `SLOTS` from the member's recorded slot names (the original
  `SLOT_SCHEME` choice), then calls the original `adopt_config_globals`.
- It also binds `log` to a list sink, the time origin, and the thread pool.

## Bound recipe (`contract.RECIPE`)

| Field | Value |
|---|---|
| Image side | 336 |
| Slots | 6 |
| Slices per slot | 12 |
| Channels | 3 slices per window, 10 overlapping windows (`window_starts`) |
| Field of view | 130 mm centre crop, only when 16 < round(130/px) < min(H, W); otherwise the full field |
| Slice band | 20–80%, `np.unique(linspace)` with the last index repeated as padding |
| Intensity | per-series 1st–99th percentile → [0, 1] → uint8 |
| Resize | bilinear, `align_corners=False` |
| Laterality | right knees: coronal/axial width flip, sagittal slice reversal |
| Missing slot | zero row, mask 0 |
| Aggregation | mean of sigmoid over windows; members are rank-averaged |

These fields are bound by the fingerprint receipt and the manifest fields in the
public-solution audit, and the reviewed node text implements them.

**Unbound in Git (`contract.UNBOUND`):** the member's `pixel_group.rules`
(`order`/`lat`/`slot_fallback`/`decode_fill`, native if absent) and its slot names. The
legacy rules reproduce the dominant-axis order, corner-x laterality, T1 fallback and
zero decode fill. Both rule sets and both slot schemes are tested. The runner reads the
actual values from the private manifest entry for `m_e5427d6c21.pt`.

## Prepared owning-node comparison (not run)

`scripts/newton/compare_public_dinov2_inputs.py` runs on evc31, CPU only, and refuses
anywhere else.

- **Bindings it checks:** the checkpoint SHA256 (bytes are hashed, never loaded), the
  manifest SHA256 and its `pixel_group`, the source/extraction/node digests, the release
  binding and the 2247/1156 fold-0 boundary.
- **Roster:** at most 64 fitting studies, chosen by salted SHA256 with no outcomes read.
- **Inputs:** each public rendering comes from `<dicom-root>/train_series/<study>/<series>/*.dcm`
  with `Anatomical_Plane` from the series CSV. It is compared with the current
  `CompetitiveStudyDataset` (16 slices, 384 px, 6 series).
- **Export:** only `compare.sanitize`d aggregates leave the node. They cover the physical
  FOV (public crop per slot vs current full field), slot occupancy per slot and
  series overlap, duplicate-series slots, distinct slices per slot, the
  geometry-ordered fraction, laterality source and mirroring counts, finite range
  quantiles, and refusals by exception type.

## Public member inference route for the next end-to-end test

1. Read the manifest entry and run `configure(namespace, pixel_group)`.
2. Run `render_study` to get a uint8 `[6, 12, 336, 336]` row and its mask.
3. Run the original `predict_member(model, cache, mask, idx, dev, 336)`. It uses 10
   windows of `[6, 3, 336, 336]`, autocast on CUDA, and the mean of sigmoids.
4. The model is the reviewed `Model`/`SlotHead` with DINOv2-small (`transformers`
   `AutoModel` from the attached config) and CLS + mean-patch pooling. `m_e5427d6c21.pt`
   is loaded with `weights_only=True`, `ck["model"]`, strict. Then the fingerprint is
   checked at the unchanged 2e-3 tolerance.

Dependencies: the exact source, the private checkpoint and manifest, the DINOv2-small
config/weights, torch 2.14, transformers, pydicom with pylibjpeg for compressed
transfer syntaxes, numpy and pandas.

## Evidence boundary

The CPU tests use synthetic DICOM. They prove that the callable renderer equals the
original batch path, and they check geometry, ordering, laterality, slot, band, crop
and window behaviour. They are not real-MRI parity, GPU parity or Kaggle accuracy. The
public high-score wrappers are ensembles with shared ancestry, and their titles are not
submission receipts, so no 0.95 single-model claim is made.
