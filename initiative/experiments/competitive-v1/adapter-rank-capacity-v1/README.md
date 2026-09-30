# Solo DINOv3 rank-capacity comparison

The bounded rank-64 real-MRI recovery pilot **865556 COMPLETED 0:0** on
September 30. Full/delta/stop/resume state and predictions match; all 120
residual projections learned and the official base stayed frozen. The matched
full ten-epoch production **865578** was staged/submitted once and is RUNNING
on evc31, one H100/two CPUs. See the actual
[launch and budget audit](../../../audits/dinov3-rank64-production-launch-v1/).
The completed rank-32 reference scored 0.8620940794 internal macro AUC on
1,156 development studies and 0.769 on Kaggle. Rank 64 has no accuracy result yet.

Preparation-time statements below remain historical; actual acceptance and
launch receipts supersede their dispatch/storage status. Frozen scientific
payload bytes and manifests are unchanged.

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

`pilot-files.json` binds the exact configs, script and verifier source. CPU staging on September 30 verified all five files, scientific code and
accepted data/fold/cache-index bytes at the unique control path. The exact
receipt is in `initiative/audits/dinov3-rank64-stage-native-profile-v1/`. Large
weights/environment/cache archive bytes are reverified inside the allocation.
Recheck the staged bytes, live jobs and projected quota before a single dispatch
intent; do not repeat staging into this existing directory.
The script also checks its bound inputs inside the allocation. Both current
Newton GPU slots are occupied; do not queue this as a third node. Use a newly
available slot only after authoritative terminal job review and fresh storage
acceptance. Do not reuse output paths or retry uncertain dispatches.

A separate role-bound private publisher and single coordinator are now
implemented: `scripts/hf/publish_adapter_a64_epoch.py` and
`scripts/hf/dispatch_adapter_a64_epoch.py`. Twenty focused tests check the actual
factory provenance and reject wrong-rank, whole-series, all-data, diagnostic,
source, membership, runtime and schedule payloads. Full QA passes 1,192 tests.
Every production epoch must satisfy the exact recipe before its private
publication intent; original rank-32 publishers remain byte unchanged.

`production-prepared.sbatch.txt` and `production-files.json` bind the exact
one-H100/two-CPU/32-GiB ten-epoch recipe and publisher. They are not staged or
dispatched. Its guard requires the actual rank-64 pilot recovery receipt,
terminal 0:0 dispatch acceptance and budget proof. The additional conservative
6.008569-GiB budget for ten delta snapshots, live/atomic writes and deployment
copies is **not accepted** against the current quota. A fresh budget must
preserve 20 GiB and 5,000 files; do not credit future cleanup.
The current rank-32 publisher must not be reused for a rank-64 payload. The
prepared ten-epoch config is not a launched training run. Every completed
production epoch must later be privately uploaded once and fully download/hash
verified. Actual T4 deployment and hidden-scale feasibility are separate gates
before any worthwhile competition submission.
