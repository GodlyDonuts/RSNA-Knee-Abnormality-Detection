# Research handoff to Codex — 2026-10-12

Audience: the agent operating `main` (Codex) and any future session. This condenses the
research/analysis produced on PR #3 and PR #4 into an ordered set of recommendations
and, just as importantly, acknowledges the defects the 2026-10-06 PR review
(`research-pr-review-20261006-v1`) already found in the packaged arms. Nothing here
asks to repeat a completed job, touch a protected source, or change an admitted
recipe — every item is a *prepared candidate* requiring the repo's normal
admission/funding path.

## 1. The single highest-leverage experiment is the label-source ablation

Codex's own review already prioritized this ("Prioritize plain
public-versus-accepted-v2 fitting-label ablation on current
rank32/native384/full10recipe"). The reasoning:

- All in-house evidence says the encoder/recipe is not the bottleneck — a copied
  public recipe scored 0.942 while our best solo is 0.804, and the biggest
  remaining lever is what the model is supervised on.
- Gold-58 arbitration (PR #4 `GOLD-ARBITRATION.md`) shows the two label sources
  are **not interchangeable — they win on different targets**:
  - v2 protocol-tracked review wins meniscus, synovitis, and ligament targets
    (public's tear-only extractor misses degenerative morphology — e.g. meniscus
    recall ~0.78 public vs ~0.89 v2 on gold).
  - Public LLM labels win the severity-defined targets (mild effusion,
    osteophytes, small Baker's cysts, early OA): v2's severity bar collapses
    recall to 0.41–0.67 vs public 0.82–1.0 on gold positives.
- Prediction before the experiment: the ablation will show a per-target split,
  not a uniform winner. If it does, a per-target blend (v2 on its 8 winning
  targets, public on the 4 severity ones) is the natural next arm — see §3
  caveat before trusting it.

Ready-made inputs on PR #4: SHA-bound mirrors and fold-bound overlays
(`label_source_v1` kind), emitted run configs under
`initiative/experiments/competitive-v1/run-configs/`, and the preregistered
analysis/decision rules in `ABLATION-ANALYSIS.md`. Note main has already
selectively integrated the plain public-label contrast — the additional overlays
(v21 variants, graded-u05, blended) live only on the PR branch.

## 2. Encoder initialization is the second lever — domain-matched MRI weights

Once the label question is answered, the cheapest real architecture-level change
is swapping the encoder initialization for an MRI-domain-pretrained one.
Ranked by cost/benefit (details in PR #4 `knee-ssl-pretraining-v1` design doc):

1. **MRI-CORE ViT-B/16** (Apache-2.0, `mazurowski-lab/mri_foundation`) — decoded
   checkpoint verified: DINOv2-teacher structure, `backbone.*` ViT-B/16, 768-dim,
   `patch_embed.proj.weight` (768,3,16,16), `pos_embed` (1,197,768), native 224px.
   Direct drop-in for our 2D extractor: strip `teacher.`/`backbone.` prefixes,
   interpolate pos_embed to 384px, discard the DINO head. Cheapest possible probe
   of the domain-init thesis — same head, same fold, one config change.
   Caveat: knee presence in its Duke-110K corpus is unverified.
2. **OrthoDiffusion** (arXiv:2602.20752) — knee-specific 3-branch 3D diffusion
   model self-supervised on 15,948 unlabeled knee MRIs; public weights verified
   downloadable (axial/coronal/sagittal ~553MB each on
   huggingface.co/lanstat0123/orthodiffusion). Strongest domain match but
   architecturally invasive (volumetric, per-view) — only after probe 1 fails.
3. **Own-corpus DINO pretraining** on the pinned knee manifest (24,386 series,
   already SHA-bound on PR #4) — day-scale GPU commitment; see blocker fix in §3.
4. **Triad** (generic 3D MRI foundation model, MIT) — same invasiveness as 2.
5. **KGNet** (TMI 2026 multi-view knee graph) — code-verified hard prerequisite:
   8-class bone segmentation maps we do not have. Do not schedule until a
   segmenter exists.

## 3. Corrections required on the PR #4 arms before they are dispatchable

The review found real defects in my packaged work. Acknowledging them here so
nobody dispatches the broken versions:

- **`entropy_max` masked-cell arm**: the term subtracts BCE(p, 0.5), which pushes
  logits *away* from uncertainty instead of toward it — the loss drives the
  opposite of the stated intent. Fix the sign (add the term, or maximize
  per-cell entropy directly) before any GPU admission.
- **`label-source-blended` overlay**: drifts 418 cells on the 8 targets it claims
  to retain from v2 — needs a cell-level equality audit against the v2 release
  before it can be called a controlled blend.
- **`canonical-sides` arm**: a horizontal flip is only a physical left/right
  mirror on axial (and some coronal) planes — on sagittal it flips A/P, which is
  not anatomically equivalent. The artifact must be re-derived per-plane (flip
  only planes where the horizontal axis is the L/R axis) or re-scoped to
  axial-only.
- **SSL manifest**: `ssl_series_manifest.parquet` is not filtered against the
  protected/outer evaluation roster — pretraining on outer studies would leak.
  Rebuild it excluding the outer roster before use.
- **EMA teacher**: the current `ema_consistency` implementation deep-copies and
  serializes the whole dense base and lets EMA touch the frozen base — it must
  be scoped to the adapter/trainable parameters only, or it breaks both memory
  and the frozen-base invariant.
- **All 26 run configs freeze the encoder with no adapter** — so none of them
  can actually exercise a domain-init probe; any encoder-init arm needs the
  LoRA/adapter path consistent with the current rank-32 recipe, or the
  comparison confounds init with trainable-parameter count.
- **Gold-informed arms cannot claim independent gold evaluation** — the blended
  overlay and the tier ordering used gold-58 to choose what to keep, so gold-58
  can no longer score them unbiasedly. Treat them as upper-bound probes only
  and evaluate on the standard outer folds.

## 4. Recommended execution order (matches owner's single-model-first direction)

1. Plain public-vs-v2 label ablation on the current recipe (prepared on main).
2. Winning-source + v2.1 severity overlays if the split result confirms.
3. MRI-CORE init probe on the winning label source (adapter path, not frozen).
4. Canonical-sides after per-plane re-derivation (probably axial-only).
5. Resolution 384→448 once the cache economics are known.
6. Masked-cell losses only after the entropy-sign fix.
7. Ensembles stay deferred until the single-model ladder is exhausted.

## 5. What I did NOT verify

- No GPU: none of these arms have an actual fit. All claims are code-level and
  fixture-verified on CPU only.
- Kaggle API auth from my VM returns 401 — no submission-side verification.
- MRI-CORE knee coverage, OrthoDiffusion's 3D input contract beyond the file
  listing, and whether gold-58's mild-severity convention actually matches the
  hidden test distribution are all open.
