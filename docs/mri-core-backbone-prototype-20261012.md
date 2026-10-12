# MRI-CORE backbone prototype — 2026-10-12

Issue #7. Research-only, opt-in, CPU-fixture-validated. No weights downloaded, no
GPU/Kaggle/cluster job, no protected gold or confirmation values read. Software tests
establish loader/adapter/resume contracts only; they say nothing about accuracy.

Code: `scripts/research/mri_core_prototype.py`; tests: `tests/test_mri_core_prototype.py`.

**Issue #9 update (2026-10-12).** The prototype forward is an intentionally
reconstructed DINOv2 CLS-token alternative. It is **not** the upstream deployed forward
and upstream forward/input parity is **not accepted** (section 1b). Original evidence
below is preserved; corrected statements are marked "corrected (issue #9)".

## 1. Source identity: verified, previously-verified, unknown

Primary-source access (github.com, arxiv.org) was blocked by this session's network
allowlist; read-only access was requested and not granted before this PR. Nothing below
is newly verified from primary sources this session.

Issue #9 session: read-only access to github.com, codeload.github.com,
raw.githubusercontent.com, api.github.com and arxiv.org was requested again and was still
not granted (TLS connection refused by the allowlist) when this update was pushed. Rows
marked "owner-verified" come from the owner's read-only check reported in issue #9; this
session did not fetch or hash those files.

| Item | Status | Value / evidence |
|---|---|---|
| Repository | prior evidence only | `https://github.com/mazurowski-lab/mri_foundation` (PR #4, commit `029d668d`, `initiative/experiments/competitive-v1/knee-ssl-pretraining-v1/README.md`) |
| Pinned upstream commit | owner-verified (issue #9) | `0f4f452614013873f8bb58a1fd613dbe36b58aba`, <https://github.com/mazurowski-lab/mri_foundation/tree/0f4f452614013873f8bb58a1fd613dbe36b58aba>. Loader now requires exactly this `source_revision`. File bytes/hashes not recorded. |
| Paper | owner-verified (issue #9) | arXiv:2506.12186 abstract (<https://arxiv.org/abs/2506.12186>): >6M slices, >110K volumes, 18 body locations. Paper data/methods sections not read. |
| License | owner-verified README statement (issue #9) | Pinned README states Apache-2.0. LICENSE file text not read by this session. |
| Checkpoint file | decoded 2026-10-06 (PR #4) | `MRI_CORE_vitb.pth`, 436 MB, Google Drive id `1nPkTI3H0vsujlzwY8jxjKwAbOCTJv4yW` linked from the repo README |
| Checkpoint SHA-256 | prefix only | `0ca90aeba09568f2…`; the full 64-hex digest was not recorded in Git. The loader requires the full digest explicitly, checks the prefix and rehashes the file. |
| Layout | decoded 2026-10-06 | `{"teacher": {...}}`; `backbone.*` ViT-B/16, width 768, `patch_embed.proj.weight (768,3,16,16)`, `pos_embed (1,197,768)` = 14x14 grid + CLS (native 224 px), plus `dino_head.*` |
| Family | decoded 2026-10-06 | DINOv2 teacher. **Not DINOv3**: no registers, no RoPE, fused `attn.qkv`, `blocks.N.*` names. The prototype records `model_family: dinov2_vit` and refuses `dinov2_*`/`dinov3_*` labels. |
| Exact upstream commit | superseded | Was unknown; now pinned above (owner-verified). |
| Layerscale / `mask_token` / unchunked blocks / MLP (not SwiGLU) / LN eps 1e-6 | **assumed** (DINOv2 ViT-B defaults) | Original text: "The checkpoint loader fails closed if any key or shape differs, so a wrong assumption cannot load silently." **Corrected (issue #9):** that holds only for assumptions visible as keys/shapes (layerscale, mask token, chunking, fused QKV, SwiGLU widths, registers). LN eps, GELU form, attention scale, QKV split order, interpolation method, CLS handling and normalization are behavioural and load silently when wrong; `test_layout_checks_do_not_detect_behavioural_assumptions` demonstrates this for bilinear-vs-bicubic and eps. |
| Input normalization and channel contract | **unknown** | Prototype offers only `imagenet_mean_std` with 1->3 gray replication and always records `input_normalization_source_verified: false` and `input_normalization_upstream_contract: unknown`. The pinned README reportedly calls `model.image_encoder` directly on [0,1] slices (owner-verified), so ImageNet mean/std is **not** established as the source contract; see 1b. |
| Declared pretraining population | prior evidence only | ~6M 2D slices, 18 body regions, Duke-110K corpus (PR #4 README / PR #6 handoff) |
| Knee coverage in pretraining | **unknown** | Not verified |
| Overlap with RSNA competition/OAI/our studies | **unknown** | Not verified; cannot be excluded |

Before any GPU use: fetch the repo at an exact commit, record LICENSE text, README weight
link, preprocessing code (mean/std, intensity scaling, channels, resolution) and the
paper's data section; record the full checkpoint SHA-256 from the existing acquisition
(do not re-download if the 2026-10-06 copy is still bound).

### 1a. Checkpoint acquisition receipt (issue #9 search)

Searched without re-downloading: every branch of this repo (`git grep` for
`0ca90aeba`, `MRI_CORE_vitb`, the Drive id), the PR #4 SSL design `provenance.json`
(covers the SSL series manifest only), this VM's filesystem, and the originating PR #4
Devin session (event search returned HTTP 403 to this automation). The only record is
the PR #4 README text at commit `029d668d` ("436 MB, sha256 `0ca90aeba09568f2…`",
Drive id `1nPkTI3H0vsujlzwY8jxjKwAbOCTJv4yW`). No download receipt, byte count, full
digest or storage location is recorded. **Full SHA-256, exact byte size and current
location of the 2026-10-06 copy: unknown.** No digest is invented; the loader still
requires the full 64-hex value supplied from a real receipt.

### 1b. Upstream path vs reconstructed CLS alternative

Two distinct forwards; do not conflate them.

| | Upstream original path (owner-reported, issue #9; not code-read here) | This prototype (`forward_variant: reconstructed_dinov2_cls_token_alternative_v1`) |
|---|---|---|
| Weight conversion | `models/sam/build_sam.py` converts DINOv2 teacher backbone weights into a SAM image encoder | `strip_teacher_backbone_split_fused_qkv_v1` into a DINOv2-style ViT |
| Deployed module | `models/sam/modeling/image_encoder.py` | `MRICoreViT` (`scripts/research/mri_core_prototype.py`) |
| CLS position token | removed | kept; CLS token prepended |
| Patch-position resize | bilinear | bicubic, `align_corners=False`, no antialias, at load time |
| Output used | SAM image-encoder output (exact shape/neck behaviour not code-read) | final-LayerNorm CLS, 768-d |
| Input | README: `model.image_encoder` called on [0,1] slices | [0,1] gray, replicated to 3 channels, then ImageNet mean/std |
| Normalization | **unknown** until the actual inference/training path is traced (dataset transforms and any preprocessing inside/around the encoder) | ImageNet mean/std (unverified choice) |

Background, not evidence: in the original `segment_anything` package pixel normalization
lives in `Sam.preprocess`, outside `image_encoder`; whether the mri_foundation fork keeps
this and whether its training path applies it is exactly what must be code-traced.

Consequences:

- The prototype is a legitimate *alternative* use of the same checkpoint tensors, but
  results from it must not be described as the MRI-CORE upstream forward or as
  reproducing upstream features.
- `reference_forward` in the tests shares every prototype assumption (CLS kept, bicubic,
  eps 1e-6, exact GELU, ImageNet normalization). It is regression coverage, not
  independent original-model parity.
- Provenance now records `upstream_forward`, `forward_variant`,
  `upstream_forward_equivalence: not_established`,
  `upstream_reference_parity_accepted: false` and
  `layout_validation_scope: keys_shapes_dtypes_finiteness_only`.

### 1c. Upstream-reference parity gate: NOT ACCEPTED

Required: a small CPU test that imports the pinned upstream `build_sam.py` /
`image_encoder.py` (no weights), feeds the same synthetic tiny teacher state through
upstream conversion and through this loader, and compares (i) converted position
embeddings, (ii) per-block activations for the shared ViT trunk and (iii) the final
encoder output under the README input contract. It could not be written or run because
the upstream source is unreachable from this session; writing it against an assumed API
would not be independent. The gate stays explicitly unaccepted until a session with
read-only github.com access records the pinned file bytes (SHA-256 per file) and runs it.
Any accepted upstream-faithful variant must be a separate `forward_variant`, not a silent
change to this one.

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
  Corrected (issue #9): the relevant upstream reference is MRI-CORE's own conversion,
  which (owner-reported) drops the CLS position and resizes bilinearly; this bicubic
  CLS-kept resize is an intentional alternative, not equivalent to it (section 1b).
- **Output.** Final-LayerNorm CLS, width 768 (DINOv3-L path: 1024). Spatial patch
  features, neighbor-3 channels, and MRNet initialization are unsupported (fail closed).
- **Adapter graph.** `rs_lora_v1` `ResidualLinear` (reused unchanged) on
  `blocks.{i}.attn.{q,k,v,proj}` for all layers and `blocks.{i}.mlp.{fc1,fc2}` for the
  last `ffn_last_layers`. Installed names and parameter count are cross-checked;
  `lora_B` is zero so the initial output equals the unadapted encoder exactly.

Tests (30, CPU, ~3 s; the fused reference shares the prototype's assumptions, see 1b):
pinned-revision binding; a demonstration that same-shape behavioural changes (bilinear
positions, LayerNorm eps) load silently and change outputs; fused-reference equivalence at native/down/up-sampled grids;
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
