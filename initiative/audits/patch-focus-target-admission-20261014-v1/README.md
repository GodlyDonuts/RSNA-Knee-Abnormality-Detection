# Patch-focus evc40 target CPU admission (prepared, not run)

Issue #84. Separately versioned route `scripts/newton/admit_patch_focus_target_inputs.py`
(`source_patch_focus_target_evc40_CPU_admission_v1`). The original owning-node
`admit_patch_focus_inputs.py` is unchanged and still refuses evc40.

Root alone stages/dispatches one fresh standalone 1 CPU / 0 GPU allocation on evc40 (never
908454/899766/899733/916757) with `--root /tmp/rsna-patch-focus-target-admission-20261014-v2`.
Root adds exclusively inside that funded root: `code/`, `source.json` (whole-body; must bind
every `rsna/` module, the original admission/wrapper/recipes, this admission and the two
reviewed audits below), `recovery-source-manifest.json`, `encoder-spec.json` and
`target.json` (effective declaration: status `root_declared_patch_focus_target_evc40_CPU_admission`,
SHA256 of the existing `funding-overlay.json`, `overlay-restored.json` and the relay
`target-verified.json`, the exact actual runtime identity, the SAME 16 MiB cap, liabilities
338320442039 B, reserve >= 20 GiB, zero GPU/checkpoint/shared bytes, `GPU_funded: false`).
Existing proofs and the overlay are never rewritten.

The admission refuses unless all of these hold, then writes `target-CPU-accepted.json` once
(exclusive create, hashes/counts only):

- placement/runtime: node evc40, `SLURM_CPUS_PER_TASK=1`, no GPU env/CUDA; actual
  `sys.prefix`/Python/package versions equal the declaration and the prepared environment;
- overlay: `/tmp/rsna-public-label-repeat-inputs-20261012-v1/overlay.json` regular, one link,
  0400, 1057379 B, exact SHA c8fd2e86..., inode unchanged since the restoration proof; private
  funding/restoration proofs agree with `patch-focus-target-overlay-restoration-20261014-v1`;
- budget: overlay plus every byte already in the root plus the 1 MiB receipt fit the same
  16 MiB cap, and fresh free space covers liabilities + reserve + remaining cap;
- cache: private relay receipt equals `dense-triplet-cache-relay-20261014-v1/actual-transfer-verified.json`
  (index 84985528.../manifest 09ea7bc6.../merged receipt d2c0ad74... lineage); exact
  3404-file layout, single-link 0400 members unchanged since the relay receipt; no bulk rehash.
  The unrelayed producer `merged-cache-verified.json` is neither required nor forged (refused if present);
- release/folds: exact shared studies/release-receipt/fold bytes of the original binding
  instead of the timed-out full archive observer; full 2247/1156 split, leakage boundary,
  supervision and cohort equal the original receipt adb931d5...;
- exact 16 fitting + 16 outer public tensor replays, official base via original stat binding
  (or one whole-byte rehash if only this node's mount device differs), bitwise matched
  initial values/logits/RNG and backward-only cloned-mask routing gradients.

No optimizer step, GPU, transfer, overlay write, protected label or accuracy claim. GPU
full/stop1/resume remains separately funded by root after an actual accepted receipt.
