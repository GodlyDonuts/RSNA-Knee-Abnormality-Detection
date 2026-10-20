# Duplicate-acquisition input failure and measured correction

Original raw CPU **978318** ended `FAILED 1:0` after 2:23:25. Its context consumer
**978414** then ended `FAILED 1:0` after 2:11:07 because its producer was terminal.
Both are absent from the live queue. The terminal-only continuation owner 57158
stopped without attempting any successor; its old policy admitted only TIMEOUT.
No training job or accepted labels were affected.

Read-only CPU **979523** reproduced the original `KeyError` at `cases.py:80` from
the retained 59,108,186-byte orphan cache. That study has 264 frame occurrences,
132 canonical frames and four acquisitions containing only aliases. The renderer
constructed acquisition ranks from canonical representatives, then attempted to
look up metadata for every duplicate occurrence, including acquisitions absent
from that canonical roster. This is a confirmed software input-assembly error.

The correction derives each occurrence's metadata from its already chosen
canonical representative. Only canonical rows are consumed by tokens/regions,
so their metadata is unchanged. Context center selection, masking, triplet
neighbors, targets and support weights retain their original definitions.
No invented rank, dropped series, relabeling or prediction fallback was added.

Actual owning-node CPU **979537** reproduced the failure and accepted the fix in
19.745 seconds, without source downloads or original-file modification. The
failing case now produces both 24-window contexts, all 2,112 canonical regions,
finite supports and complementary masks. Three previously successful real cases
at context caps 8 and 24 retained every tensor/support byte and RNG state.
The structural canonical-row invariant and 42 scoped tests additionally cover
successful-domain parity, multiple matrices, cached descriptors and unequal-pixel
alias rejection. This is an input correction, not an accuracy result.

- Original cases SHA: `c19e3d45b6573dbd9c631f67e2794493e9e9f4a06f6da49df93f16edb6d93cf8`.
- Corrected cases SHA: `5f7c81fd270c072fa16ed486735049bb72070a704304b6ee1adc2c4f6d4f3d09`.
- [Actual regression](actual-regression.json), SHA `946911a47151af80225cf28921876e88e089c35c86e4c11061ca141bbe2e463e`.
- [Original terminal records](actual-original-terminals.json).
- [Read-only original failure](actual-diagnosis.json).

The original 506 verified raw records plus 16 reused originals (**522 studies**),
retained partial source buffers, orphan cache and existing context prefix stay
in place. Recovery must continue only missing work, preserve the distinct old
and corrected renderer provenance, and admit the actual regression before any
successor dispatch. It must not relabel old archive hashes as new source.

Local full-fitting owner **59538** was stopped while waiting, after confirming
zero phase intents, no transport children and no training jobs. Its original
initialization/model/optimizer recipe remains preserved. The
[handoff receipt](actual-old-fitting-owner-stopped.json) permits a distinct
owner that understands the two honest renderer versions; it does not authorize
restarting the old owner or repeating remote work.
