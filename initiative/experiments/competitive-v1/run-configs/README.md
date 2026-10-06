# Verified run configs for the packaged arms

Every `*.train.json` here is the production recipe
(`configs/competitive/dinov3-campaign.json`
`train_config_without_encoder` + the pinned `dinov3-large` encoder block)
plus exactly one arm field, emitted by
`scripts/research/emit_arm_configs.py` and deserialized through
`TrainConfig(...).validate()` at emission time — a config that fails
validation never lands on disk. `tests/test_arm_configs.py` re-validates
them on every run.

## How to run an arm

Artifact-bearing arms bind the sha256 inside the config; the artifact path
travels via the matching CLI flag:

```bash
python -m rsna.competitive.train \
  --config initiative/experiments/competitive-v1/run-configs/<arm>.fold<N>.train.json \
  --output <run_dir> \
  [--fitting-supervision <overlay.json>]   # label-source-* and graded-u05
  [--canonical-sides <table.json>]         # canonical-sides-*
```

| config | arm | CLI artifact flag |
|--------|-----|-------------------|
| `base-v2-control` | v2 baseline control | none |
| `label-source-public.fold{0,1,2}` | public `llm_labels_v4_blend` labels | `--fitting-supervision .../llm_labels_v4_blend.fold<N>.overlay.json` |
| `label-source-blended.fold{0,1,2}` | v2 on 8 targets + public soft on 4 severity targets | `--fitting-supervision .../blended_v2plus_public.fold<N>.overlay.json` |
| `label-source-v21.fold{0,1,2}` | v2 + all v2.1 positive flips (~3,600/fold) | `--fitting-supervision .../v21_severity_rereview.fold<N>.overlay.json` |
| `label-source-v21pn.fold{0,1,2}` | v21 + explicit-negative flips (~290/fold) | `--fitting-supervision .../v21pn_severity_rereview.fold<N>.overlay.json` |
| `label-source-v21hc.fold{0,1,2}` | v2 + pass-2-only flips (gold-prec 0.75 tier, ~2,200/fold) | `--fitting-supervision .../v21hc_severity_rereview.fold<N>.overlay.json` |
| `graded-u05.fold{0,1,2}` | `uncertain` cells supervised at 0.5 | `--fitting-supervision .../graded-states-u05.fold<N>.overlay.json` |
| `canonical-sides.fold{0,1,2}` | deterministic canonical-R mirror | `--canonical-sides .../canonical-sides.fold<N>.json` |
| `masked-cell-em005` | ESA entropy-max on masked cells, λ=0.05 | none (config-bound) |
| `masked-cell-consistency` | mean-teacher EMA consistency on masked cells (α=0.999, λ=1.0 ramped 20%) | none (config-bound) |
| `pooling-max` | max-pooled series aggregation vs per-diagnosis attention | none (config-bound) |
| `slice-smooth-smmil` | SmMIL adjacent-slice score smoothness, λ=0.1 | none (config-bound) |

A sha256-mismatched artifact is rejected at load — the binding is the point.
`fold` inside each config matches the overlay's roster binding.

## Regenerate

```
PYTHONPATH=. python scripts/research/emit_arm_configs.py
```

Re-run after rebuilding any overlay: the script re-hashes the artifacts and
re-validates every config.
