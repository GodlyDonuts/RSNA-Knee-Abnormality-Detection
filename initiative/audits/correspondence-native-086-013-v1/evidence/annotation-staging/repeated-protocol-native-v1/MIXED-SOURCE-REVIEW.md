# Corrected-source native correspondence review

The remaining two repeated-protocol candidates were reviewed from the corrected pinned Hugging Face source after Kaggle returned HTTP429. All original partial attempts and83 successfully fetched official048 files remain preserved. No further Kaggle requests were made during the corrected-source native runs.

| Case | Complete source files / series | Retained official originals | Corrected HF files awaiting fresh official byte confirmation |
|---|---:|---:|---:|
| 048:12 | 359 /10 | 83 | 276 |
| 086:13 | 300 /10 | 0 | 300 |

Every file in each complete study was fully decoded in the new run, including the retained83 originals. Full SHA-256, size, inventory Study/Series/SOP UIDs, archive CRC, native sample fingerprint and original dimensions match. Every decoded ImagePositionPatient, ImageOrientationPatient and PixelSpacing matches the previously bound source-specific full-cache metadata. The15-field source geometry/encoding record is retained per slice, but only fields available in cache are asserted to match cache. CRC/index equality alone is not represented as fresh official whole-byte confirmation for mirror-sourced objects.

Both complete runs used at most two requests and a20GiB reserve. The corrected source is0b5e410288232f570c69ba4f44723356e6ae6ab7. `source-origin-plan.json` binds the official correction receipt and each source record states its origin. Corrected mirror downloads use inventory-bound LFS content hashes; they do not infer revision membership merely from a mutable branch. No original source, cache, clinical annotation or gold value was changed.

Each case has four contact sheets with three native slices per series at approximately25/50/75 percent ordered positions. Native stored samples are displayed with their whole-series native minimum/maximum, with centered padding and no image resize, crop, rotation or mirror. The viewer may reduce large sheets; this is disclosed. Thirty displayed panels per case are a visual sample, not full-volume visual inspection. Root receives panel bindings before this agent's independently frozen visual observations.

`audit_native_mirror_v1.py` is the exact execution script. It consumes the frozen six-case source/header handoff, existing full-cache metadata and accepted corrected-source receipt. The original official attempts are preserved separately in `case-048-12` and `case-048-12-paced-v1`, with explicit interruption/status records. No failed request is silently counted as verified coverage.

`verify_retained_mixed_sources.py` rechecks all659 retained full-file hashes, sizes, CRCs and UID headers, including83 original official source references, without network or another native pixel decode. Its result also records all-slice hashed identifier counts without real-world identity inference. `deferred-official-source-confirmation.json` names exactly576 paths still awaiting authoritative byte confirmation.

The completed075 case is separate and immutable:337 original Kaggle sources and33 sampled panels with independent visual review. Its initial unpublished display window was superseded explicitly after a rescale-versus-native display mismatch; originals and source proof were never changed.
