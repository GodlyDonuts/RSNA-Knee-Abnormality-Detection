# Retain the checkpoint's complete384 field

Prospective October17 protocol. The exact attributed0.949parent checkpoint declares
res384; its executed source uses INPUT_RES320 and crops the cached384 image by
32pixels on each side. Test one fixed alternative: retain the full384 field,
44%more area relative to320, same140mm source crop/96slices/94windows/two anatomical
views, unchanged learned weights/rank output. This is additional original image
coverage, not upscaling or a larger backbone. The relative-position grid also
changes with input size; this is not a claim that only border pixels affect output.

First inspect32deterministic admitted fitting cases without labels/model weights:
original320 hashes must reproduce, both views must be bit-equal to the central
320square of the full384 tensors, and the outer band must contain original
nonzero MRI signal. Record aggregate thresholds0/8/32 on actual present slices.
Signal alone is not a lesion or an accuracy gain. No new pixel archive is needed.

Only after actual CPU acceptance prepare a separate offline T4 qualification.
Require strict identical learned parameter tensors at both sizes, functional384
model forwards, exact CPU/GPU input values and original320control reproduction
against its actual saved logits on the original32engineering cases. Models may
fail to accept the new grid; do not infer compatibility from checkpoint metadata.
No clinical/Gold/confirmation labels or independent internal OOF claim.

An accepted distinct384 public-MRI/CSV candidate may then use a fresh Kaggle slot
against the already scored0.949320reference. No strength/resolution sweep, no
new weight upload, and no alteration of healthy full-parent training or existing
entries. It could hurt by diluting central evidence or shifting position bias.
Fresh optional GPU pushes must preserve the primary training budget and actual
per-version receipts. Public execution never substitutes for a hidden result.
