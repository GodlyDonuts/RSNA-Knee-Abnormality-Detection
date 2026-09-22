# Preparation checkpoint: 491 accepted corrections

Git commit: `fec046253168083514749226987dda5b6c2505ce`. Git tree: `77321d30b1e6970c062e0a4c64791bc03b203638`. Exact-commit CI run 35693648375 passed.

The pinned registry contains 491 accepted target corrections. Report extraction, source correspondence and final training/resume/inference checks remain separate readiness requirements. This publication does not close a QA gate. **Training readiness is false.** Gold remains evaluation-only. The 491 milestone passed 462 local tests; exact-commit CI is separately verified before publication.

The complete manifest contains 10020 Git files. Every path, mode, size, Git blob ID and SHA-256 is verified from the immutable base plus the delta, including entries omitted from the earlier base archive. Actual peer and root restoration proofs are separately bound. Verification streams both archives without extracting another complete tree.

Required base: dataset revision `54949cd5a575ea7027a4a57916e772936a98d38b`, `preparation-checkpoints/1792f01ed3daab757c5ddb9bf5733ffe8d999f91/checkpoint.tar.gz` and its `manifest.json`. The delta is not standalone. After verifying safe regular-file paths, start from the base, apply the manifest deletion list, overlay the delta and verify the complete result against the pinned target manifest. Use the accompanying checkpoint_delta.py streaming verifier.

Original train/test image trees, the five official metadata CSVs, the root dataset card, and the 576 explicitly tracked corrected-mirror source files remain unchanged. The historical deferred official-byte confirmation for those mirror files is not resolved by this checkpoint. This is a private preparation checkpoint, not a training release or diagnostic accuracy claim.
