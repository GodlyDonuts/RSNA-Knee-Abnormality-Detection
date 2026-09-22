# Competitive image path implementation

This implementation adds an offline global-embedding baseline without changing the
accepted data release. It is ready for controlled CPU wiring tests and Newton
benchmarking. The synthetic tests do not validate pretrained accuracy, H100 execution,
competitive scores, spatial-token models or 3D encoders. A separate bound local
[ConvNeXt acceptance receipt](../initiative/audits/competitive-pretrained-local-check-v1/convnext-actual-cpu-receipt.json)
records an actual complete pinned pretrained load, 224-pixel CPU forward and
finite head backward pass. Its temporary weight file was hash-verified and evicted
afterward to preserve the storage reserve.

## Encoder contract

`create_encoder(spec)` returns `(encoder, embedding_dim, provenance)`. Encoders
accept floating grayscale `[N,1,H,W]` in `[0,1]`; the production adapters repeat
channels and apply ImageNet RGB mean `[0.485,0.456,0.406]` and standard deviation
`[0.229,0.224,0.225]` internally. Do not normalize a second time. There is no random
crop, channel conversion library, automatic weight download, remote model code,
or automatic checkpoint-key conversion.

Production specs require these exact fields:

```json
{
  "architecture": "dinov2_b14_reg",
  "config_path": "local-weights/dinov2/config.json",
  "config_sha256": "<64 lowercase hex digits>",
  "weights_path": "local-weights/dinov2/model.safetensors",
  "weights_sha256": "<64 lowercase hex digits>",
  "transformers_version": "5.6.1",
  "source_id": "facebook/dinov2-with-registers-base",
  "source_revision": "<40 lowercase hex digits>",
  "license": "apache-2.0",
  "state_dict_format": "safetensors"
}
```

This schema example contains intentionally invalid placeholder pins. Use real
reviewed source manifests. Paths are relative to the working directory. Returned
provenance excludes local paths and includes source revision, config/weight hashes,
architecture, pooling and preprocessing. Source identities/licenses are supplied
manifest assertions; the offline loader does not independently prove upstream
license eligibility or approve any access terms.

Supported architectures use explicit Transformers 5.6.1 constructors:

- `dinov2_b14_reg`: 768 dimensions, 12 layers/heads, 14-pixel patches and four
  register tokens; the final normalized CLS token is used.
- `dinov3_b16`: 768 dimensions, 12 layers/heads, 16-pixel patches, four storage
  tokens and a 3,072-wide MLP; the final normalized CLS token is used.
- `dinov3_l16`: 1,024 dimensions, 24 layers, 16 heads, 16-pixel patches, four
  storage tokens and an ordinary 4,096-wide GELU MLP. This is the web ViT-L
  variant, not SwiGLU L+ or the satellite model with untied normalization.
- `convnext_tiny`: depths `[3,3,9,3]`, widths `[96,192,384,768]`; normalized global
  spatial mean. The default `model_class` is `ConvNextForImageClassification`:
  strict-load the complete HF classification state, then retain its `convnext`
  backbone. Explicit `model_class="ConvNextModel"` instead requires a strict
  backbone-only state. The unused classification head never enters the optimizer.

The adapters require dimension multiples of 14 for DINOv2, 16 for both DINOv3
variants and 32 for ConvNeXt; matched
224×224 works for all three. Local single-file safetensors and plain PyTorch tensor
state dictionaries are supported. Nested training checkpoints, upstream native
DINO state layouts and sharded checkpoints require a separately reviewed explicit
conversion, not silent prefix stripping. Tensor values must be finite and every
state key/shape must load strictly. Optional dependencies are
`transformers==5.6.1` and `safetensors==0.7.0`; torchvision and timm are unnecessary.
GPU binary compatibility still needs validation on the actual node.

`{"architecture":"diagnostic_tiny","embedding_dim":16}` creates a random small
CPU encoder. Its provenance explicitly says `diagnostic_only=true` and
`pretrained=false`. The trainer must require a positive diagnostic study limit;
this factory alone does not authorize production use of that diagnostic model.

Primary architecture references:
[DINOv2 register backbone](https://github.com/facebookresearch/dinov2/blob/main/dinov2/hub/backbones.py),
[DINOv3 backbone](https://github.com/facebookresearch/dinov3/blob/main/dinov3/hub/backbones.py),
[DINOv2 HF config](https://huggingface.co/facebook/dinov2-with-registers-base/blob/main/config.json),
[ConvNeXt HF config](https://huggingface.co/facebook/convnext-tiny-224/blob/main/config.json).
The installed Transformers forward implementations were also inspected: DINO
returns the normalized CLS token as `pooler_output`; ConvNeXt returns a
layer-normalized spatial mean. The DINOv3 model repository was gated during this
work; no access terms were accepted and no pretrained state was downloaded.

## Data and geometry

`CompetitiveStudyDataset(studies, series, cache, slices=16, side=224,
max_series=6, augment=False, raw_split="train_series")` consumes only frames already
admitted by the release/fold validator. It preserves the existing series selection
policy, effective planes and optional sampling tiers. The five series features
are three plane indicators, fluid sensitivity and fat suppression; sampling tier,
report text, laterality and identity are not model inputs.

The dataset checks existing cache version, study/series identities, resolution and
source revision through the frozen `StudyDataset` initializer. It additionally
binds source path fingerprints, unique ordered SOP membership/hash counts, physical
geometry and a fixed sidecar digest. Every selected NPZ is read once into bytes,
checked against its exact SHA256/size, and then decoded with pickling disabled.
This adds per-epoch hash I/O. Full inventory/release binding remains the caller's
responsibility; these local checks do not replace the complete image audit.

For each series, at most `min(slices,n_slices)` distinct native-order indices are
selected. Remaining slots have zero images, positions and masks. Deterministic
sampling spans the ordered stack. Augmentation only changes series ordering and
samples one index from each disjoint integer slice bin; it does not flip, crop or
rotate images. The existing aspect-preserving padded square is resized as a whole.

Positions are `[centered_projected_mm / 100, known_flag]`. Projection uses the
unit cross product of the stored row/column orientation. Series with complete,
strictly ordered patient geometry get flag one, including a single physical slice
with centered distance zero. An instance-number fallback gets `[0,0]`; index order
is never represented as a physical distance. Invalid claimed geometry fails.
Centering removes cross-series absolute position; this is not spatial registration.

## Study model and tests

`CompetitiveStudyNet(encoder, embedding_dim, pooling="attention", hidden_dim=128,
encode_chunk_size=64)` consumes images `[B,S,K,1,H,W]`, features `[B,S,5]`, boolean
series/slice masks and positions `[B,S,K,2]`. It returns twelve logits in the frozen
schema order. Only genuine valid slices reach the encoder. Slice embeddings plus
positions feed target-specific gated attention, followed by target-specific gated
series attention and twelve target heads. `pooling="max"` is the fixed comparator;
unused attention parameters are frozen. DINO mask-token parameters remain frozen
even when the backbone is unfrozen because this unmasked task never consumes them.
A fully frozen encoder remains in evaluation
mode when the study model enters training mode.

Chunking bounds each encoder call; it does not release all retained training
activations or implement activation checkpointing. The initial implementation uses
global embeddings only. It has no image segmentation, regional tokens, registration,
3D adaptation, cached embedding store or image-derived diagnostic labels.

Tests exercise finite one-slice predictions, strict state restoration, invalid-input
rejection before encoding, NaN padding invariance, genuine slice coverage, gradients,
source/cache tampering, physical/unknown positions, augmentation replay, and exact
normalization. Small random versions of the three real Transformers architectures
execute without pretrained downloads. These probes establish software behavior,
not equivalence to running actual licensed pretrained weights on Newton.


## DINOv3-L access and validation status

The current account's authenticated request for the official pinned HF config
returned HTTP 403 `GatedRepo` on September 22. The server explicitly states that
the existing access request is awaiting review by the repository authors.
Public metadata identifies revision
`ea8dc2863c51be0a264bab82070e3e8836b02d51`, a manual gate, license category `other`,
and a 1,212,559,808-byte safetensors checkpoint with SHA256
`dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179`.
These are remote metadata assertions; neither the config bytes nor complete weights
have been verified locally. No executable Large manifest is fabricated from them.
Access and the applicable agreement must be resolved before that checkpoint is staged.

The architecture contract was checked against Meta's public `dinov3_vitl16`
constructor and Transformers 5.6.1's explicit DINOv3 config/model implementation.
The loader validates ordinary GELU MLP, RoPE base 100/rescale 2, four registers,
query/value/output/MLP biases, absent key bias and LayerNorm epsilon 1e-5.
LayerScale initialization is preserved from the checksummed source config; every
learned LayerScale tensor is subsequently required by strict state loading.
The public upstream constructor initializes these scales to 1e-5; the HF config
class default is 1.0, so an initialization scalar is not mistaken for a learned
checkpoint value. No source config is rewritten to change either value.

Tests cover the complete 24-layer topology on the meta device without allocating
large weights, plus an actual random one-block model with Large width, head count
and MLP through the 1,024-dimensional encoder and twelve-target study head. The
latter includes backward gradients and preserves the unused mask token outside
the optimizer. Neither test establishes pretrained Large inference or accuracy.
Global CLS features remain the launch scope; local patch pooling is deferred.
