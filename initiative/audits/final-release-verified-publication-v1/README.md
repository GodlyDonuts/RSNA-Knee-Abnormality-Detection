# Verified report-supervision data release

Git commit: `f41879a33b38e47f0cbd96025edac7e2176a0a2f`. Git tree: `aed8cda7cbb394bf665319a52073ab8926a16a10`. Exact-commit CI run 35745491694 passed.

The separately bound final data release has status `verified_release`; its final-input diagnostic acceptance passed, all three semantic QA gates are resolved for that release, and all 569 prior accepted corrections remain unchanged. Release receipt: `initiative/releases/report-supervision-v1/verified_release.json` SHA-256 `6453a8ff31373c40556dc857e15fa0025f122195388a2b46c81ecd9cff0b8833`. Diagnostic acceptance: `initiative/audits/final-input-training-acceptance-v1/acceptance.json` SHA-256 `cd680364f174224d278ea37be80dd335edf7c8bff8fc90066c0648c569053067`.

Acceptance verifies two completed short diagnostic epochs, exact CPU resume, masked zero gradients, all 58 gold-study inference outputs and all 3 released test outputs. These are software and data acceptance checks, not a competitive accuracy claim. Gold remains evaluation-only. Competitive training strategy, tuning and full training remain separate work.

**Transport metadata scope:** the unchanged delta helper retains its historical `training_ready=false` field in the manifest and streaming proofs. That unchecked legacy field does not certify or assess data readiness. The final-data eligibility statement above is supported only by the separately pinned release and acceptance records.

The complete manifest contains 11145 Git files. Every path, mode, size, Git blob ID and SHA-256 is independently reconstructed by the actual peer and root from the immutable base plus delta. The two peer proof names are aliases of one execution; the root proof comes from a separate actual execution. No full extra tree is extracted.

Required base: dataset revision `54949cd5a575ea7027a4a57916e772936a98d38b`, `preparation-checkpoints/1792f01ed3daab757c5ddb9bf5733ffe8d999f91/checkpoint.tar.gz` and its `manifest.json`. This delta is not standalone. Verify safe regular-file paths, apply the explicit deletions and overlay delta precedence, then verify all final manifest identities with the accompanying unchanged checkpoint_delta.py.

This is a private six-object checkpoint prefix. Original train/test image trees, five official metadata CSVs, root dataset card and 576 tracked corrected-mirror files remain unchanged. Their preserved origin limitations are unchanged; metadata preservation is not fresh official-byte confirmation.
