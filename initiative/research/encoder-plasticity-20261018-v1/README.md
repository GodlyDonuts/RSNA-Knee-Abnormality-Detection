# Encoder adaptation: actual epoch-1 to epoch-3 movement

October 18, 2026. CPU job **956497** completed a new aggregate measurement on
existing private uniform/influence checkpoints. No images, clinical targets or
protected values were read. No GPU fit, old checkpoint rehash or bulk transfer
was performed. Original checkpoint hashes/private revisions are in
[actual-result.json](actual-result.json).

| Role | Encoder relative L2 movement | Head relative L2 movement |
| --- | ---: | ---: |
| Uniform | 0.0425918% | 0.613031% |
| Influence | 0.0419113% | 0.625251% |

The encoder has 72,856,264 parameters; over 99.998% changed in each role.
Batch-normalization buffers stayed unchanged. Encoder/head peak learning rates
were 3e-6/3e-5. This is the adapted 73M CoAtNet branch, **not DINOv3**.

This measures **epoch 1 to epoch 3**, not initialization to final. Weight-space
movement is not feature-space movement or an accuracy metric. Large movement
can destroy a good representation; small movement can be sufficient. Different
parameter scales also limit encoder/head comparisons. The result establishes
that the encoder updates, and motivates an adaptation-strength experiment; it
does not establish a learning-rate root cause.

## Next comparison

A proposed single-factor comparison increases encoder peak LR from 3e-6 to
3e-5 while retaining head LR 3e-5, original parent initialization, influence
proposal, 3 epochs/1458 updates, all 3887 fitting studies, labels/masks, teacher
mixture, frozen BN buffers, seed, optimizer and inference. Start from the same
parent, not the old epoch-3 model. The existing influence final score 0.949 is
the public reference; no new gain or independent internal validation is claimed.
The existing exact-source validators must not silently accept the new recipe.
**No higher-LR training job or notebook has been dispatched.**

Fresh authenticated quota CPU956499 reports 98,291.4286632 seconds used, zero
reserved, minimum 108,000 and total allowance 162,000 seconds. This leaves
9,708.5713368 seconds against the minimum and 63,708.5713368 against the reported
total. The existing policy reserves against the smaller bound. Three 6000-second
epoch caps alone exceed that conservative remainder, before outstanding serving
work. No paid quota was purchased or enabled. Resolve actual usable capacity or
use a separately admitted H100 route before claiming this full experiment fits.
Healthy finding training and existing scoring owners remain intact.

## Source and focused checks

`drift.executed.py.txt` preserves the exact executed source SHA from the actual
receipt. `drift.py` is its subsequently formatted implementation; do not claim
its new text hash was the executed hash. `analyze_once.py.txt` preserves the
original one-shot dispatch script for provenance, not authorization to rerun.
Five focused synthetic cases cover numerical aggregation and malformed inputs;
these checks are engineering evidence, not classifier performance.
