# Native384 fixed-final result

Original job 867367 completed 0:0 after ten epochs and 2,810 contiguous finite optimizer updates. Scientific C9, accepted masks/folds, pinned official DINOv3-L weights and the 2,247-training/1,156-development study split stayed unchanged. All ten immutable epochs are privately published and fully download/size/SHA verified; no production restart occurred.

Fixed-final native384 internal macro AUC is **0.863055822603059**, against the predeclared strict uniform224 reference **0.8606037925461784**: delta **+0.0024520300568806608**. This is weak-label internal evaluation, not Kaggle. Paired whole-leakage-component bootstrap used 4,000 original draws; 2,836 were defined and 1,164 undefined. The conditional 95% interval is **[-0.0142284307496993, 0.018149981198996416]**. The gain is not established. Undefined rare-target draws are retained rather than redrawn.

| Target | Uniform224 | Native384 | Delta | Observed positives |
|---|---:|---:|---:|---:|
| ACL | 0.883832 | 0.882092 | -0.001740 | 85 |
| MCL | 0.641121 | 0.561839 | -0.079281 | 2 |
| Medial Meniscus | 0.863234 | 0.868415 | +0.005181 | 412 |
| Lateral Meniscus | 0.735145 | 0.759759 | +0.024614 | 149 |
| Medial OA | 0.976388 | 0.975603 | -0.000784 | 23 |
| Lateral OA | 0.770036 | 0.779556 | +0.009521 | 9 |
| PF OA | 0.865109 | 0.848492 | -0.016617 | 19 |
| Effusion | 0.884501 | 0.916791 | +0.032290 | 9 |
| Synovitis | 0.980165 | 0.976860 | -0.003306 | 121 |
| Baker's | 0.891361 | 0.911431 | +0.020070 | 3 |
| Contusion | 0.904887 | 0.937608 | +0.032721 | 61 |
| Fracture | 0.931467 | 0.938224 | +0.006757 | 2 |

Native initial learned-state equality was not collected and independently reconstructed caches differ; this is not a strict causal resolution-only trial. Some targets have very few positive examples (MCL has two), so their apparent changes are especially fragile. No best-epoch selection or protected confirmation/gold was used. Keep the uniform recipe as the established reference and repeat a promising resolution effect on a separately matched seed/fold before promotion. A useful Kaggle final-checkpoint comparison may proceed without claiming a clinical gain.

Replay: `PYTHONPATH=. PYTHONNOUSERSITE=1 artifacts/qa-env/bin/python -B initiative/audits/native384-fixed-final-v1/recompute.py`. It verifies the original source/config/roster/runtime/terminal/checkpoint/prediction bindings and independently replays masked AUC and paired uncertainty. `collection.json` binds all small original artifacts; no checkpoint/cache bulk was recollected.

Native final checkpoint SHA: `24cf7a0c936eeeb99e9a5c83009d1e0d59792fb249978aeaf41231c34ef3371a`. Native epochs 9/10 private revisions: `ce090a223195bced0b2c24b83e854c218e519237` and `18beabe1e72a9ffbc0e47ed2a8f7f07a5a3020d2`. Never repeat these publications.
