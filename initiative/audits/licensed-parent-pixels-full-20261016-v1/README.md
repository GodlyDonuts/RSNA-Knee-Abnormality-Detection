# Full fitting input-cache production: four CPU shards

Prepared after the actual32CPU Kaggle/node gate passed. Frozen source parent
493c7b0d, codec7bdaf41a; no rendering/quantization/resolution change. All3887
admitted fitting UIDs come from original1,261,832byte index5a1a4c77 and
UIDhash0485d1b1. Four disjoint fixed ranges0:972/972:1944/1944:2916/2916:3887
cover the original population exactly once. Source data/protected masks unchanged.

Each private CPU notebook reuses original32 compressed members via the accepted
pixel source; only new studies are decoded/encoded. New members preserve full
96x384uint8volumes/masks and94two-view input hashes. Empty/decode-error cases
are explicitly counted/retained; they cannot be silently admitted for GPU fitting.
At most14GiBoutput/shard plus2GiBscratch reserve, no GPU or competition entry.
The future trainer mounts immutable shard outputs read-only. Large originals
remain on authorized private cluster/Kaggle storage; no MRI/index toMac/Git/HF.

Before production, independently fund on evc39 after all607078528963prior
required bytes:64GiBnewcache,12GiBfuture recovery/epoch/publication envelope,
1GiBoptional parent-weight envelope and128MiBcontrols, zero shared growth or
deletion credit. These envelopes do not accept a numerical trainer or authorize
a duplicate parent copy. Exact actual checkpoint size/source and GPU recovery
remain required. Reuse the existing original32 on evc33 and new complete shards
on evc39; no second transfer of the original32pixel archive is needed.

Eighteen focused codec/extraction/sharding tests passed, Ruff/mypy3clean. Full
production has NOT yet been staged/dispatched at this preparation snapshot.
