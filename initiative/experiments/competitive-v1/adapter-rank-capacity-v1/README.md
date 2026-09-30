# Solo DINOv3 rank-capacity comparison

Rank-64 sampled-input recipes and a bounded recovery pilot are prepared. **No
GPU job has been reserved or dispatched.** The completed rank-32 reference
scored 0.8620940794 internal macro AUC on 1,156 development studies and 0.769
on Kaggle. Rank 64 has no training or accuracy result yet.

The prepared reference and challenger differ only in adapter rank. They keep
the same official DINOv3-L base, 120 projection locations, alpha 8 with
`alpha/sqrt(rank)` scaling, dropout 0.05, frozen original weights, accepted
labels/masks/fold, seed, 16 slices, six selected series, 224 inputs from the
256 cache, attention head, effective batch eight, optimizer and ten-epoch
schedule. The rank-32 reference differs from the historical A32 recipe only
in the exact frozen-base delta storage format. Native resolution, whole-series
coverage, neighboring context, new labels, teachers and ensembles are separate
comparisons.

| Capacity | Encoder adapters | Study head | Total trainable |
| --- | ---: | ---: | ---: |
| Rank 32 | 10,223,616 | 203,020 | 10,426,636 |
| Rank 64 | 20,447,232 | 203,020 | 20,650,252 |

Local `TrainConfig` validation and the actual complete production DINOv3 graph
enumeration passed for both ranks. The complete study graph also reproduces
the head counts and 417 frozen encoder entries, including the two normalization
buffers. These checks use the meta device and allocate no model weights. They
do not establish real gradients, GPU throughput, learned weights or MRI accuracy.
The original and replayable verification scripts and exact config hashes are
preserved here. To recheck without writing into the archive:

```bash
PYTHONPATH=. PYTHONNOUSERSITE=1 artifacts/qa-env/bin/python \
  initiative/experiments/competitive-v1/adapter-rank-capacity-v1/verify-config-and-graph.py.txt
```

Using the same seed does not guarantee identical random head initialization
when the adapter parameter count changes. This is a fixed-recipe capacity
comparison. Repeat seeds and independent development folds before attributing
a broad gain to rank. Report all twelve target AUCs, component support and
paired rare-component sensitivity; reject promotion based only on lower
training loss or one sparse target. The outer pool is a development set;
reserved confirmation and official evaluation remain untouched.

## Allocated acceptance before production

`pilot-prepared.sbatch.txt` prepares one H100/two CPUs/32 GiB, bounded to two
hours, with the unchanged scientific runtime `c9fcc7e`. The separately hashed
acceptance verifier is external to that clean checkout. Its parameter check
now measures the actual learned graph against factory-verified provenance,
instead of hardcoding the rank-32 count. Regressions cover both production
ranks and refuse false capacity, missing projections or a trainable base tensor.
The running rank-32 jobs and their pinned verifiers remain unchanged.

The pilot uses 16 training and 16 outer studies for two epochs/four updates.
It compares full storage with uninterrupted delta training, intentionally stops
a separate delta run after one update, resumes it, and requires equal model,
optimizer, RNG, progress and prediction state. All 120 residual projections
must learn, all official base tensors must remain unchanged, and a strict
real-MRI reload must pass. Its tiny predictions are engineering evidence only;
the preserved diagnostic is never competition eligible.

`pilot-files.json` binds the exact configs, script and verifier source. Actual
staging must check those bytes, the complete cache/environment/data bindings,
live jobs and projected byte/file quota **before** a single dispatch intent.
The script also checks its bound inputs inside the allocation. Both current
Newton GPU slots are occupied; do not queue this as a third node. Use a newly
available slot only after authoritative terminal job review and fresh storage
acceptance. Do not reuse output paths or retry uncertain dispatches.

Production also needs a role-bound private publisher and a fresh storage plan
covering ten rank-64 delta snapshots, live/atomic writes and deployment copies.
The current rank-32 publisher must not be reused for a rank-64 payload. The
prepared ten-epoch config is not a launched training run. Every completed
production epoch must later be privately uploaded once and fully download/hash
verified. Actual T4 deployment and hidden-scale feasibility are separate gates
before any worthwhile competition submission.
