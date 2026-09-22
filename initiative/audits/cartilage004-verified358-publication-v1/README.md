# Preparation checkpoint: 358 accepted corrections

Git commit: `3d3b135f3dbc44763f78c60e950d074565850f49`. Exact-commit CI run 35682228129 passed; 462 local tests passed.

Both original review passes cover all 4,407 reports. The registry contains 358 accepted target corrections and 18 enforced study exclusions from report-derived training loss. Three global semantic QA gates and final training/resume/inference validation remain pending. **Training readiness is false.** Gold labels remain evaluation-only.

This checkpoint represents all 8955 files of the pinned Git tree. Its manifest lists exact paths, modes, sizes, Git blob IDs and SHA-256 hashes. The delta includes new/modified files and files omitted by the earlier preparation archive.

The required immutable base is dataset revision `54949cd5a575ea7027a4a57916e772936a98d38b`, files `preparation-checkpoints/1792f01ed3daab757c5ddb9bf5733ffe8d999f91/checkpoint.tar.gz` and `manifest.json`. The new delta is not a standalone replacement for that base. Both archives were streamed and every resulting member verified against the complete target Git tree, without extracting another full copy.

To reconstruct, start with the verified base archive, remove the manifest's explicit deleted_paths, then overlay delta.tar.gz. Verify the complete resulting membership, every file hash/size/mode and the manifest's pinned Git identity. Never extract unverified paths or links. checkpoint_delta.py provides a streaming verification command (`verify --repo <pinned-git-repo> --target 3d3b135f3dbc44763f78c60e950d074565850f49 --base-archive <base-tar> --base-manifest <base-manifest> --output <this-directory> --expected-manifest-sha256 ff71c97af5fae320cbaa83dc97e29bdadb742f2bb5bf063a3e5b2558184c2b8d`).

The original source image trees, five official metadata CSVs and root dataset card are preserved. This is a private preparation checkpoint, not a finished training release or a diagnostic accuracy claim.
