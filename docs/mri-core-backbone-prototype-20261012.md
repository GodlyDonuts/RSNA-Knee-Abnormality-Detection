# MRI-CORE backbone prototype — 2026-10-12

Issue #7. Research-only, opt-in, CPU-fixture-validated. No weights downloaded, no
GPU/Kaggle/cluster job, no protected gold or confirmation values read. Software tests
establish loader/adapter/resume contracts only; they say nothing about accuracy.

Code: `scripts/research/mri_core_prototype.py`; tests: `tests/test_mri_core_prototype.py`.

## 1. Source identity: verified, previously-verified, unknown

Primary-source access (github.com, arxiv.org) was blocked by this session's network
allowlist; read-only access was requested and not granted before this PR. Nothing below
is newly verified from primary sources this session.

| Item | Status | Value / evidence |
|---|---|---|
| Repository | prior evidence only | `https://github.com/mazurowski-lab/mri_foundation` (PR #4, commit `029d668d`, `initiative/experiments/competitive-v1/knee-ssl-pretraining-v1/README.md`) |
| Paper | prior evidence only | arXiv:2506.12186 (same README) |
| License | prior evidence only | Apache-2.0 (same README). Re-verify the LICENSE file at an exact commit before any use. |
| Checkpoint file | decoded 2026-10-06 (PR #4) | `MRI_CORE_vitb.pth`, 436 MB, Google Drive id `1nPkTI3H0vsujlzwY8jxjKwAbOCTJv4yW` linked from the repo README |
| Checkpoint SHA-256 | prefix only | `0ca90aeba09568f2…`; the full 64-hex digest was not recorded in Git. The loader requires the full digest explicitly, checks the prefix and rehashes the file. |
| Layout | decoded 2026-10-06 | `{"teacher": {...}}`; `backbone.*` ViT-B/16, width 768, `patch_embed.proj.weight (768,3,16,16)`, `pos_embed (1,197,768)` = 14x14 grid + CLS (native 224 px), plus `dino_head.*` |
| Family | decoded 2026-10-06 | DINOv2 teacher. **Not DINOv3**: no registers, no RoPE, fused `attn.qkv`, `blocks.N.*` names. The prototype records `model_family: dinov2_vit` and refuses `dinov2_*`/`dinov3_*` labels. |
| Exact upstream commit | **unknown** | Loader requires a 40-hex `source_revision`; none is pinned here. |
| Layerscale / `mask_token` / unchunked blocks / MLP (not SwiGLU) / LN eps 1e-6 | **assumed** (DINOv2 ViT-B defaults) | The checkpoint loader fails closed if any key or shape differs, so a wrong assumption cannot load silently. |
| Input normalization and channel contract | **unknown** | Prototype offers only `imagenet_mean_std` with 1->3 gray replication and always records `input_normalization_source_verified: false`. |
| Declared pretraining population | prior evidence only | ~6M 2D slices, 18 body regions, Duke-110K corpus (PR #4 README / PR #6 handoff) |
| Knee coverage in pretraining | **unknown** | Not verified |
| Overlap with RSNA competition/OAI/our studies | **unknown** | Not verified; cannot be excluded |

Before any GPU use: fetch the repo at an exact commit, record LICENSE text, README weight
link, preprocessing code (mean/std, intensity scaling, channels, resolution) and the
paper's data section; record the full checkpoint SHA-256 from the existing acquisition
(do not re-download if the 2026-10-06 copy is still bound).

## 2. Prototype contract

- **Isolation.** Lives in `scripts/research/`, outside `rsna/`. Inference binds
  `code_sha256` over `rsna/**/*.py` (`rsna/competitive/infer.py`), so every existing
  checkpoint's binding is unchanged. `create_encoder` and `TrainConfig.validate()` still
  reject this architecture (tested); `TrainConfig` still admits adapters only for
  `dinov3_l16`. Production wiring is a later, separate decision.
- **Spec (exact fields).** `architecture=mri_core_vitb16`, `weights_path`,
  `weights_sha256`, `source_revision`, `architecture_config`, `image_side`,
  `input_normalization`, optional `adapter`. The config must equal the bound ViT-B/16
  values (768/12 layers/12 heads/3072 MLP/patch 16/native grid 14).
- **Key mapping `strip_teacher_backbone_split_fused_qkv_v1`.** Top level must be exactly
  `{"teacher"}`; `teacher.backbone.*` must equal the expected key set and shapes;
  only `dino_head.*`/`ibot_head.*` and `backbone.mask_token` are discarded (recorded in
  provenance). Missing, unexpected (`register_tokens`, chunked `blocks.0.0.*`,
  `student.*`), wrongly shaped, non-float or non-finite tensors raise before any module
  is built. Fused `qkv` is split along dim 0 in q, k, v order into separate `attn.q/k/v`
  linears; a fused-QKV reference forward written from the DINOv2 equations matches.
- **Position interpolation `cls_kept_grid_bicubic_align_corners_false_no_antialias_fp32_v1`.**
  Done once at load from the native grid to `image_side/16`; the forward rejects any
  other input side. Upstream DINOv2 interpolates at runtime with a 0.1 offset and
  `scale_factor`; this load-time size-based resize is not claimed bit-identical to it.
- **Output.** Final-LayerNorm CLS, width 768 (DINOv3-L path: 1024). Spatial patch
  features, neighbor-3 channels, and MRNet initialization are unsupported (fail closed).
- **Adapter graph.** `rs_lora_v1` `ResidualLinear` (reused unchanged) on
  `blocks.{i}.attn.{q,k,v,proj}` for all layers and `blocks.{i}.mlp.{fc1,fc2}` for the
  last `ffn_last_layers`. Installed names and parameter count are cross-checked;
  `lora_B` is zero so the initial output equals the unadapted encoder exactly.

Tests (28, CPU, ~3 s): fused-reference equivalence at native/down/up-sampled grids;
interpolation invariants; 8 checkpoint-layout and 10 spec fail-closed cases; production
digest-prefix binding; zero-residual identity; only `lora_A/B` trainable; frozen base
byte-identical after AdamW updates; `lora_B` moves; unfreezing rejected; gradient
checkpointing gradients match; and an exact full vs stop-after-1/resume trainer run with
`checkpoint_format=frozen_base_delta`, dropout 0.1, comparing every checkpoint tensor and
the outer prediction CSV bytes.

## 3. Capacity and cost (architecture arithmetic, not measurements)

| | DINOv3-L/16 (current) | MRI-CORE ViT-B/16 |
|---|---|---|
| Pretraining | web-scale natural images (LVD) | ~6M MRI slices (declared) |
| Width / depth / heads | 1024 / 24 / 16 | 768 / 12 / 12 |
| Base encoder parameters | ~303M | 85,817,856 (incl. unused mask token) |
| Tokens at 384 px | 576 patches + CLS + 4 registers | 576 patches + CLS |
| Adapter rank 32, FFN last half | 10,223,616 (FFN last 12) | 3,833,856 (FFN last 6) |
| Rank 85, FFN last 6 (parameter-matched) | — | 10,183,680 (-0.4%) |
| Approx. forward GFLOPs / 384 px slice | ~385 | ~111 (~3.47x fewer) |

The current 10,426,636 trainable total = 10,223,616 adapter + head parameters.

Two-T4 / nine-hour envelope (`initiative/audits/kaggle-runtime-budget-20261005-v1`:
32,400 s hard limit, 28,800 s target with margin, 2x T4 16 GiB, 4 CPU, ~29 GiB RAM,
offline). The measured reference is the 1,300-study private fitting-scale FP32 resident
run of checkpoint `85bfc1ce420f` (`initiative/audits/kaggle-resident-fullscale-v1/actual-terminal.json`):
18,303.6 s pipeline, 0.0710 studies/s, 2.36 GB peak GPU allocation per device. A ViT-B
encoder at the same input does fewer FLOPs per slice, so the same pipeline is expected to
stay at or below that envelope, but decode/IO share is unknown and nothing here measures
it. Any MRI-CORE candidate still needs its own end-to-end T4 runtime admission before a
competition slot.

## 4. Proposed matched experiment (not run; needs normal admission/funding)

Question: at the current recipe, does the MRI-CORE ViT-B/16 backbone with an rs-LoRA
adapter fit better or worse than the current DINOv3-L/16 rank-32 adapter, on the same
data? This is a **backbone-package comparison**, not an initialization-only causal test:
architecture, width, depth, capacity, token layout, pretraining corpus and objective,
input normalization and adapter graph all differ together. A difference cannot be
attributed to MRI-domain initialization alone.

- Arms: (A) existing DINOv3-L rank-32 control; (B) MRI-CORE rank 85 / FFN last 6
  (adapter-parameter matched, ~10.18M); optional (C) MRI-CORE rank 32 / FFN last 6
  (rank matched, 3.83M) to see whether any gap tracks adapter capacity.
- Held fixed: fitting supervision (the original, unless the matched full-public repeat
  below changes the default first), release, masks, folds, 2,247 fitting / 1,156
  weak-development studies, native 384 px, 16 slices / 6 series, seed, effective batch,
  fixed 10 epochs / 2,810 updates, warmup/cosine, head LR, fixed-final evaluation with
  the same component-bootstrap protocol and a predeclared primary macro metric. Encoder LR
  may need a single predeclared value for the smaller backbone; do not tune it on the
  weak-development set.
- Reuse the existing T4 full/stop/resume acceptance pattern and own-H100 recovery before
  any production fit. One seed is a weak signal; promotion would need a matched replicate.
- Out of scope: protected gold/confirmation evaluation, Kaggle submission, clinical claims.

Prerequisites: primary-source verification (section 1); full checkpoint digest from the
existing acquisition; production wiring of the factory into `create_encoder`/`TrainConfig`
with an explicit inference `code_sha256` bump reviewed as a new code version.
