# Retain spatial variation before global pooling

ActualCPU936392 inspected the pinned CoAtNet architecture: it reduces each
1024×10×10feature map to one1024-vector. Reapplying the original head reconstructs
the original output exactly. Distinct maps can have identical global means; a
synthetic focal-versus-uniform example confirms our new summaries distinguish
such maps. Neither observation proves useful disease information was lost.

The hypothesis is that some useful focal/spatial information can be recovered
without another full CNN fine-tune by observing its frozen pre-pool maps. A new
representation retains **all1024channels** for seven summaries: four quadrant
means minus global mean, central6×6mean minus global mean, spatial population
standard deviation, and spatial maximum minus mean. Coordinates refer to the
actual network input view, not a proven anatomical/common physical frame.
No anatomy registration, tissue localization, novelty or accuracy is assumed.

All reductions are FP32; storage is explicitlyFP16 with finite/overflow checks
and scaled rounding error<=0.001. Any future training and serving must use that
same representation. A forward-pre-hook returnsNone and never replaces the
original input. Each observer has explicit begin/end boundaries so qualification
or unrelated forwards cannot mix into an MRI bag.

First bounded engineering test: original32allowed fitting studies, original
licensed parent7e5315da/source493c7b0d, two94-window views. Compare original
pixelsCPU/GPU, hooked versus unhooked original features/logits within the same
forward, and old cached probabilities (unchanged error bound0.003). Within-pass identity is
bit-exact. Original32failed a cross-run globalfeature0.001bound; the distinct
two-study diagnostic found bit-exact observer identity but1.51–2.79%cross-run
feature differences and <=0.000139parent probability differences. Root numerical
cause is unproven. The corrected representation saves global FP32features and
parent logits from the SAME forward as regional features, and never substitutes
old global embeddings. Old feature error/bit-exact view counts are diagnostic,
not an acceptance criterion for the new coherent input. Check the first actual feature-map
chunk CPU/GPU regional scaled error<=0.0001 on bothT4s. Retain all regions/channels,
stream one private tar,4pendingCPUtasks/2workers/2queue,128MiBprobe cap. No labels,
reports, Gold, confirmation, optimizer, model fitting or submission.csv. Actual
admission/full cache/trainer/accuracy must each be demonstrated separately.

The existing global feature cache is complete. A bounded reassignment may use
128MiBof its unused4GiBreservation only after actual retained-original byte/stat
accounting proves the parent caches still fit the remaining bound. This preserves
all originals and the aggregate funded threshold; no deletion credit. It upgrades
the earlier64MiBregional graph/probe reservation rather than hiding a new liability.
No full regional cache or new GPU trainer is funded or dispatched by this probe.

This uses ideas from [CLAM](https://www.nature.com/articles/s41551-020-00682-w)
(region-level weak supervision) and [TransMIL](https://proceedings.neurips.cc/paper/2021/hash/10c272d06794d3e5785d5e7c5356e9ff-Abstract.html)
(correlations among instances). Those papers study pathology, not proof of a knee
MRI benefit. This proposed coarse representation is not either paper's exact
architecture. The practical decision is whether actual numerical/size/runtime
acceptance supports a separately funded regional-head experiment; no clinical
promotion follows from a feature probe.
