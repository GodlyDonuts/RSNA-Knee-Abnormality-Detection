# Complete official-source duplicate proof

Fresh authenticated Kaggle downloads confirm complete duplicated image content between studies ending **99725523560254732094432244556623376711** and **99926624968240681772735102303531613209**. Each has five series and 144 slices. All 288 original DICOMs are preserved under `official-verification-v1/native/`.

The complete 50,996,372-byte download was bounded to two requests concurrently with a 20 GiB disk reserve. All 288 full-file SHA-256 hashes, sizes, archive CRC32 values and actual Study/Series/SOP UID triplets match the corrected inventory and official archive index. All 288 native decodes match the frozen mirrored native fingerprints, original dimensions and positions. All samples are nonconstant. Five one-to-one series pairs contain 144/144 exactly equal full decoded matrices, native shapes/dtypes and per-slice geometry/encoding dictionaries. All 20 off-diagonal series combinations have zero native fingerprint overlap in the frozen characterization.

The exact compared geometry/encoding fields are ImagePositionPatient, ImageOrientationPatient, PixelSpacing, SliceThickness, SpacingBetweenSlices, Rows, Columns, SamplesPerPixel, BitsAllocated, BitsStored, HighBit, PixelRepresentation, PhotometricInterpretation, RescaleSlope and RescaleIntercept. Numeric vectors are compared as parsed floats; thickness, spacing-between-slices and rescale values are compared as parsed string representations. This is exhaustive coverage of those fields for all 144 paired slices, not an assertion that every DICOM header element is identical.

All 288 whole-file source hashes and SOP UIDs are distinct. The duplicated matrices therefore reside in different DICOM files. Hashed PatientID and FrameOfReferenceUID values differ across the two studies. These anonymized identifiers do not establish real-world patient identity or the upstream reason for duplication. The proof establishes this pair's duplication is present in official Kaggle files; it makes no corpus-wide claim that all other mirrored sources are byte-identical to Kaggle.

Both studies, original source files and all duplicate edges must remain preserved. Treat the pair as one leakage-isolation component in data splits; no cross-study deletion or clinical label inference is made. Final training release depends on the separate complete release gates.

## Evidence and reproduction

- `official-verification-v1/official-verification.json` binds corrected inventory, official ZIP index, the previously frozen cache characterization, execution script and exhaustive results.
- `official-source-records.jsonl` within that directory records every official file's identity, full-byte checks, native fingerprint, geometry/encoding and comparison flags.
- `official-full-matrix-comparisons.jsonl` records all 144 direct full matrix and geometry comparisons.
- `verify_official.py` is the exact preserved execution script (SHA-256 679e4345515214a8eacf261eaa6ac26419ffb03fd6770db58b2a83f3143fbbb5). It accepts `--work-root`, a NEW `--output`, `--workers 1|2` and optional `--env-file`; reexecution performs fresh authenticated downloads. Credentials and signed URLs are never evidence outputs.
- `verify_official_integrity.py` rechecks all 288 preserved source SHA/size/CRC/UID bindings and all frozen result/evidence references without network access or another pixel decode. Its output lists every retained original.
- `render_official_pairs.py` generated ten native middle-slice panels (one A/B pair per matched series), using one shared linear display window per pair and centered native pixels without resizing, cropping, rotation or flipping. The panel manifest binds all selected sources and display parameters. Sampled visual inspection cannot replace the exhaustive numeric result.

Execution scripts and original evidence are immutable once their hashes are bound. New review observations or execution status belong in separate artifacts. The original `README.md` and cache-characterization handoff intentionally describe the earlier mirror/cache-only stage; this document adds the later official verification without revising that historical scope.
