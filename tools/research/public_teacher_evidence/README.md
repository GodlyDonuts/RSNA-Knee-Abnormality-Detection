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
