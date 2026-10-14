# Frozen public-teacher evidence (issue #35)

Private research mechanism. Turns the already downloaded Pilkwang v15 member
`m_e5427d6c21.pt` into fitting-only soft evidence for a solo DINOv3 student. Not
capsule packaging (#33); no downloads, jobs, uploads or submissions. Not run by the
authoring session; no accuracy claim.

## What the teacher is

- Source: exact `initiative/audits/public-dinov2-source-handoff-20261013-v1/source.notebook.json`
  (SHA `b32a9155…`). The reviewed renderer namespace (`public_dinov2_input_reference`,
  node SHA `6805a20d…`) supplies crop, slots, laterality, series-order replay, triplets,
  ten TTA windows and `predict_member`.
- Teacher nodes ([teacher_review.json](teacher_review.json)): `POOL_PARTS`,
  `SLOT_PRIOR_TABLE`, `SLOT_PRIOR_STRENGTH`, `SEED`, `SlotHead`, `Model`,
  `FINGERPRINT_TOL`, `fingerprint`, `check_fingerprint`, as original source segments.
- Replaced original calls: `build_model` (`AutoModel.from_pretrained`) by an explicit
  hash-bound DINOv2-small `Dinov2Model(Dinov2Config(...))` whose weights come only from the
  checkpoint's strict complete state; original `torch.load(weights_only=False)` by
  `weights_only=True` on SHA-verified bytes (`7223cd50…`). Original fingerprint check
  (tolerance 0.002) must pass before any MRI forward.

## Evidence exposed

The original source exposes **no attention or per-slot output**: `SlotHead.forward`
keeps `att` local and returns logits; `Model.forward` reads only `last_hidden_state`.
Collected per fitting study: ten per-window `Model.forward` logits (read-only hook) and
the original `predict_member` probabilities, which must be reproduced bitwise from those
logits, plus slot mask and input digest. Nothing else is fabricated.

## Boundaries

- Fitting roster only: the complete 2,247-UID roster must hash to `e9d38ee2…`; any other
  UID is refused. Roster file holds UIDs only; no labels/reports/gold/confirmation/
  evaluation pixels are read.
- Teacher training roster: **unknown**. No clean held-out teacher/student comparison,
  competition eligibility, clinical truth or accuracy can be claimed.
- Evidence lives apart from labels (`rsna/competitive/teacher_evidence.py`); accepted
  labels, masks and folds are unchanged. Immutable directory (`evidence.npz` +
  `receipt.json`, 8 MiB cap), reconciled on repeat, refused on any binding difference.
- Student inference needs only the student; no teacher/ensemble at inference.

## Root-owned eight-study acceptance (before any full collection)

CPU, owning node, retained parity packet; renders the same salted eight fitting studies
as `public-dinov2-real-inputs-20261013-v2`. Root supplies the captured DINOv2-small
config used by the fingerprint receipt and a UID-only fitting roster.

```bash
R=/tmp/rsna-public-dinov2-real-inputs-20261013-v2
python3 scripts/newton/collect_public_teacher_evidence.py --mode render \
  --source initiative/audits/public-dinov2-source-handoff-20261013-v1/source.notebook.json \
  --member-manifest $R/private-member-manifest.json --checkpoint $R/checkpoint.pt \
  --backbone-config <DINOv2-small config.json> --backbone-config-sha256 <its SHA256> \
  --fitting-roster <private fitting-uids.json (2247 UIDs, SHA e9d38ee2…)> \
  --dicom-root $R/original --series <train_series_descriptions.csv> \
  [--series-order <recorded {uid: [series order]} JSON>] \
  --roster-size 8 --device cpu --out /tmp/rsna-public-teacher-evidence-acceptance-v1
```

Acceptance = `status: collected` (or `reconciled` on repeat) with 8 studies, finite
evidence and the original fingerprint check passing. Only then may root run
`--mode cache --cache <full public triplet cache> --acceptance <that out dir>`, which
refuses unless the accepted receipt has the identical teacher/source/runtime binding.

## Matched research design

Precommitted in [design.json](design.json): ordinary public-input DINOv3 versus the same
input/base/seed/schedule plus `w=0.5` confidence-weighted teacher auxiliary BCE
(`teacher_evidence.combined_loss`; `w=0` is exactly the baseline). Trainer wiring is
deliberately not part of this change and needs separate review after acceptance.

## Twenty-member committee with an exact shared prefix (issue #39)

Root's inspection (`initiative/audits/public-teacher-committee-prefix-20261013-v1`,
not part of this change) reports byte-identical embeddings and blocks 0-5 across all
twenty retained v15 members (113 tensors, 11,404,416 elements, digest `f127a88b…`) and
one pixel group `bbd75640`. [committee.py](committee.py) uses this for computation only:

- `verify_prefix` re-proves it on the loaded members: same graph class and backbone
  config, frozen eval graphs without hooks, and bitwise-equal (dtype/shape/device/bytes)
  prefix parameters, buffers and the `Model` `mean`/`std` normalization buffers; sizes
  must equal 113/11,404,416. Any difference refuses; members are never partially shared.
- Each member keeps its full graph from `teacher.build_teacher` (original fingerprint
  check). A shared view is the member's own original `Model` with only `backbone`
  swapped for prefix (shared) + blocks 6-11 + layernorm (member). Original
  `Model.forward`, pooling, `SlotHead`, masks, `window_starts`, `predict_member` and
  `teacher.collect_study` run unchanged; no source math or normalization was altered.
- `SharedPrefix` reuses an output only for a bitwise-identical normalized input with the
  same shape/stride/dtype/device/autocast/grad/inference state; bounded to one study's
  ten windows and cleared before and after every study. Any prefix tensor change
  (version counter, or bytes vs the verified snapshot, checked per study) refuses.
- `Committee.collect_study(check_full=True)` compares every member's shared window
  logits/probabilities with its full graph bitwise and refuses on any difference;
  `mode="full"` is the ordinary fallback. `verify_fingerprints` requires shared and full
  fingerprints to be equal bitwise and the original check to pass.

Root acceptance (owning node, not done here): run `verify_fingerprints` and
`collect_study(check_full=True)` on the eight accepted render studies under the exact
production device/precision. If bitwise equality fails, report it and collect with
`forward=full_graph`; do not relax tolerances. Synthetic CPU tests
(`tests/test_public_teacher_committee.py`) are mechanism checks only. The real
`Dinov2Model` test runs only where `transformers` is installed.

Evidence: [`rsna/competitive/committee_evidence.py`](../../../rsna/competitive/committee_evidence.py)
stores per-member window logits/probabilities, the uniform mean and between-model std,
fitting-only and apart from labels/masks/folds, under the same immutable
write-once/reconcile rules. Its proposed **64 MiB** cap (the full 2,247-study roster does
not fit 8 MiB) is not funded: writes refuse without root's funding receipt SHA.
Committee confidence uses between-model disagreement, not across-window spread. Members
share ancestry and are not independent votes; teacher training roster stays unknown.
Collector wiring for root's receipt/manifest is deferred until that receipt is
committed. Protocol: [committee_design.json](committee_design.json).
