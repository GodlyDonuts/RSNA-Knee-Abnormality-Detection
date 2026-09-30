# Bounded original-DICOM reconstruction and measured cache differences

CPU-only **865553 COMPLETED 0:0**. It fully downloaded/hash verified **215
original DICOMs / 161,939,848 bytes** across eight metadata-selected development
series from private source revision `0b5e410288232f570c69ba4f44723356e6ae6ab7`.
Source byte/UID/pixel/order/geometry checks passed. Actual source volumes were
rebuilt at **256, 384 and 448** using unchanged scientific preprocessing
`c9fcc7e`. This is not upsampling the accepted 256 cache to claim recovered detail.

Selection used metadata, split and observation masks; no target values, reports,
reserved confirmation labels or gold labels were read. Matrix sizes range from
256 to 1024, with sagittal/coronal/axial and fluid/non-fluid-sensitive coverage.
The series are a bounded engineering sample, not a representative accuracy set.

The initial exact-equality probe **865547** failed after four exact series when
its fifth 256 reconstruction differed from the accepted cache. That failure,
source/helpers, partial outputs and dispatch are preserved unchanged under
`initial-exact-check/`. The distinct reconciliation retained the same frozen
accepted cache and quantified all eight series rather than rewriting it.

| Series | Rebuilt 256 vs accepted cache | Differing pixels | Largest unsigned 16-bit difference |
| --- | --- | ---: | ---: |
| 01 | Exact | 0 | 0 |
| 02 | Exact | 0 | 0 |
| 03 | Exact | 0 | 0 |
| 04 | Exact | 0 | 0 |
| 05 | Measured difference | 559 | 1 |
| 06 | Measured difference | 1,045 | 1 |
| 07 | Exact | 0 | 0 |
| 08 | Measured difference | 710 | 1 |

Five arrays match exactly; the other three differ by at most **1 / 65,535** in
normalized intensity. The specific kernel/rounding cause is unproven. No model
accuracy effect or clinical insignificance is established by this bound. All
selected accepted cache NPZ/metadata members were independently byte/hash
verified. The successful probe records selected tar member offsets and stops
archive traversal once its 16 required members are found. It does not claim a
new full 82.8-GB archive rehash.

`matched-center-slices.png` was rendered from only four matched central slices
with an identical intensity scale and nearest-neighbor display; full volumes
were not copied to the Mac. Actual visual inspection found corresponding
anatomy, framing and intensity, with finer visible edges in some higher-matrix
448 reconstructions. No diagnosis, label correction, lesion visibility guarantee
or competitive performance claim follows. The 320-source case is itself
upsampled at 448; the larger 512/640/1024 cases are reconstructed from their
original pixels. Rendering/plane/NPZ hashes are in `center-preview-render.json`.

Actual derived volumes remain privately on Newton:
`/home/sa305415/rsna-knee/native-source-decode-reconcile-v1/result/`.
The verified retained data were **147,307,843 bytes** before the final receipt,
within a **512-MiB** bound; the failed first control has its own separate bound.
Every output identity/hash and rebuild input is retained in `measured-differences/`.
These derived volumes have not been published to HF yet. Original DICOMs already
remain at the pinned private HF revision. A full native cache or model training
has not been launched; it requires a complete storage/runtime design and a
matched resolution-only held-out comparison. The accepted cache and all labels,
masks/folds remain unchanged.
