# Actual private Kaggle input-shape admission

CPU-only offline kernel138609817, godlydonuts/rsna-input-shape-cpu-21x3-20261014-v1
version1, was pushed once under the canonical coordinator and COMPLETED. Exact
source/private/offline/zeroGPU settings and all21 notebook plus3 Dataset inputs
were authenticated and reconciled. The actual runtime matched all20 original
shard-manifest hashes in19.51seconds without reading MRI pixels, reports, labels
or models. No competition submission or GPU reservation occurred.

This proves that this particular21/3 input shape is supported; it does not
establish a global maximum. Do not repeat the probe. Empty competition sources
were omitted as null by GetKernel: a first read-only observer rejected that
representation, then actual metadata was inspected and reconciled to empty.
The saved source was unchanged and no second SaveKernel occurred.

Kaggle documentation confirms20GB saved outputs under/kaggle/working:
https://www.kaggle.com/docs/notebooks. The new production continuation must
export its state there, preserving every immutable epoch and exact provenance.
