# Preparation checkpoint: 333 accepted corrections

Git commit: `dc806961fb99cdbaa5da2e6c9a9d3ee78e062cc4`. Exact-commit CI run 35679684402 passed; 462 local tests passed.

Both original review passes cover all 4,407 reports. The registry contains 333 accepted target corrections and 18 enforced study exclusions from report-derived training loss. Three global semantic QA gates and final training/resume/inference validation remain pending. **Training readiness is false.** Gold labels remain evaluation-only.

This checkpoint represents all 8651 files of the pinned Git tree. Its manifest lists exact paths, modes, sizes, Git blob IDs and SHA-256 hashes. The delta includes new/modified files and files omitted by the earlier preparation archive.

The required immutable base is dataset revision `54949cd5a575ea7027a4a57916e772936a98d38b`, files `preparation-checkpoints/1792f01ed3daab757c5ddb9bf5733ffe8d999f91/checkpoint.tar.gz` and `manifest.json`. The new delta is not a standalone replacement for that base. Both archives were streamed and every resulting member verified against the complete target Git tree, without extracting another full copy.

To reconstruct, start with the verified base archive, remove the manifest's explicit deleted_paths, then overlay delta.tar.gz. Verify the complete resulting membership, every file hash/size/mode and the manifest's pinned Git identity. Never extract unverified paths or links. checkpoint_delta.py provides a streaming verification command (`verify --repo <pinned-git-repo> --target dc806961fb99cdbaa5da2e6c9a9d3ee78e062cc4 --base-archive <base-tar> --base-manifest <base-manifest> --output <this-directory> --expected-manifest-sha256 456697efa1d9f85d0d3116f74bcd6130bb077f6b5f1043c4f45fc9dc920258ae`).

The original source image trees, five official metadata CSVs and root dataset card are preserved. This is a private preparation checkpoint, not a finished training release or a diagnostic accuracy claim.
