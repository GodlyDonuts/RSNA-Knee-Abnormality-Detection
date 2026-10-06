# Competitive-v1 experiment index

One table for every arm in the competitive line — what it is, what binds it,
and where it stands. Each subdirectory has the full protocol; this index is
the landing page. Release for all arms: `report-supervision-v1`; folds:
`folds/fold.csv`; outer fold: 1,156-study disjoint holdout.

Run configs for every packaged arm live in `run-configs/` — production
recipe + one arm field each, validated through `TrainConfig` at emission and
re-validated by `tests/test_arm_configs.py`; regenerate with
`scripts/research/emit_arm_configs.py`.

## Awaiting a GPU slot (artifacts packaged, loader-verified)

| arm | contrast | artifact |
|-----|----------|----------|
| `label-source-ablation-v1` | public LLM labels vs v2 reviewed states, paired per study | `llm_labels_v4_blend.fold0.overlay.json` + v2 baseline |
| `blended-source-v1` | per-target source: v2 on 8 targets, public soft probs on 4 severity targets | `blended_v2plus_public.fold{0,1,2}.overlay.json` |
| `graded-targets-v1` | v2 `uncertain` cells supervised at 0.5 vs masked | `graded-states-u05.fold{0,1,2}.overlay.json` |
| `laterality-normalization-v1` | deterministic canonical-R mirror vs no flip | `canonical-sides.fold{0,1,2}.json` |
| `masked-cell-losses-v1` (EM arm) | ESA entropy-maximization on `mask==0` cells vs ignore | `TrainConfig.masked_cell` — config-bound, no artifact |

## Design / plan only

| arm | status |
|-----|--------|
| `masked-cell-losses-v1` (consistency EMA arm) | teacher copy + ramp not built |
| `knee-ssl-pretraining-v1` | DINO self-supervised pretraining on our own corpus — GPU-infeasible locally |
| `slice-relations-v1` | opt-in relation integration exists; clinical read pending the full-fit pipeline |

## Completed / superseded

See `neighbor-channels-v1` (three-arm fixed-final comparison — paired
intervals include zero, grayscale kept as control), `patch-attention-v1`,
`native-detail-v1`, `native-resolution-profile-v1` (Nyquist: 224px fails
99.6% of series; ceiling ~448px), `effusion-review-contrast-v1`,
`stratified-slices-v1`, `whole-series-context-v1`, `patch-witness-v1`,
`adapter-rank-capacity-v1`.

## Label-quality analysis chain (2026-10-06)

`label-source-ablation-v1/` now carries the full semantics record:
`DISAGREEMENT.md` (14,206-cell queue) → `TIER0-ADJUDICATION.md` →
`OA-CONFLICT-TRIAGE.md` → `GOLD-ARBITRATION.md` (58-gold per-target split
verdict + corpus coverage table) → `V21-PROTOCOL-PROPOSAL.md` (~5,000-cell
re-review under a lowered severity bar). The blended overlay is a
gold-informed upper-bound probe under `label_source_experiment_only` — it
measures headroom, it is not a shippable label policy.
