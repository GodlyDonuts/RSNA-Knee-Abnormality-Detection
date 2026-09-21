# Complete cross-study image-content duplication

The corrected full audit's only cross-study nonconstant pixel edge is a complete
five-series image-content duplicate between:

- `1.2.826.0.1.3680043.8.498.99725523560254732094432244556623376711`
- `1.2.826.0.1.3680043.8.498.99926624968240681772735102303531613209`

Each study has five series and 144 slices. All 144 distinct nonconstant native
fingerprints occur once in each study. There are five one-to-one full-series
matches, zero unmatched slices, and zero overlap between the other 20 series
pairings. All corresponding ordered native fingerprints, native dimensions,
every recorded position, orientation, spacing, model metadata, full cached
arrays, and compressed NPZ byte hashes agree exactly.

| Matched sequence metadata | Slices in each study |
| --- | ---: |
| Sagittal, fluid-sensitive, fat-suppressed | 30 |
| Sagittal, non-fluid-sensitive, without fat suppression | 30 |
| Coronal, fluid-sensitive, fat-suppressed | 27 |
| Coronal, non-fluid-sensitive, without fat suppression | 27 |
| Axial, fluid-sensitive, fat-suppressed | 30 |

The two studies have distinct source files and identities: all 288 SOP UIDs are
distinct, and no whole-DICOM source SHA256 is shared across studies. Their cache
metadata differs only in `cache_path`, `path_fingerprint`, `series_uid`,
`slice_sha256`, `sop_uids` and `study_uid`. This establishes image-content
duplication rather than whole-file DICOM identity.

The 20 pre-existing endpoint records show one hashed PatientID per study and
five hashed frame identifiers per study. Neither type of identifier is shared
across the two studies. Both have literal `R` laterality tags and the same five
sequence descriptions. Distinct anonymized identifier values therefore did not
prevent this image duplication; they do not prove distinct real-world patients.

Keep both studies and preserve the cross-study edge. They belong in the same
leakage component, with weak/evaluation isolation applied by the existing label
release gates. This assessment does not remove either study, assign clinical
labels, or determine the original cause of duplication.

## Evidence and reproduction

`evidence/edge-characterization.json` binds the complete corrected audit, its
series index, effective inventory, source-series metadata, full recovery lineage,
baseline metadata, corrected metadata, every NPZ, the old endpoint survey and
this script. All ten corrected metadata hashes also match the frozen recovery
lineage; each equals its preserved baseline metadata with only the accepted
source revision changed.

- `inventory-pair.json` preserves all 288 exact source rows.
- `all-source-slice-bindings.jsonl` binds every ordered native fingerprint to its
  source SOP/path/hash/size, native shape, position and cached slice digest.
- `shared-native-hash-groups.jsonl` preserves all 144 shared hashes and both owners.
- `series-overlap-matrix.json` includes all 25 cross-study series comparisons.
- `complete-series-matches.json` gives the five exact full-SeriesUID matches.
- `series-summary.json` includes every recorded position and sequence feature.
- `recovery-lineage-pair.jsonl` preserves the ten original recovery records.
- `endpoint-headers.jsonl` and `technical-endpoint-identifiers.json` preserve
  the existing endpoint technical evidence and hashed identifiers.

From the recovered repository, with the original baseline metadata referenced
by the recovery lineage still available:

```bash
python ../annotation-staging/cross-study-pixel-edge-v1/characterize.py \
  --work-root .. \
  --output ../annotation-staging/cross-study-pixel-edge-v1/evidence-repeat
```

The script rejects an existing output. It reads no reports, clinical review
annotations or gold values, and performs no downloads or native DICOM decodes.
It independently hashes and reads all ten cache arrays in full. Source-native
fingerprints come from the original complete strict decode and are verified
through their preserved inventory/metadata/recovery bindings; this is not a
claim of a second native decode. The existing endpoint evidence retains its
partial-prefix limits. No visual interpretation or clinical image diagnosis was
needed for the complete numerical equivalence result.
