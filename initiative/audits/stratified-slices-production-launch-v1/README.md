## Actual start and exact uniform recovery

Stratified866716 passed actual CUDA/header/full source/cache/init and started
finite full-cohort optimizer updates on evc32. Its first two losses0.8260/0.8579
and observed encoder gradient are in `stratified-production-start/`; no accuracy
gain is claimed. Original uniform866715 FAILED2:0 in evc34's bounded driver probe
before any optimizer/header/output/checkpoint. The underlying node cause is
unproven. Distinct unchanged uniform866719 was submitted once on pilot-accepted
evc31, initially PENDING Priority. See
[the exact terminal and recovery](uniform-evc31-recovery-v2/README.md).
Keep both original failed and distinct retry records. Do not repeat either
mutation. The historical first-queue paragraph below is superseded by these
actual outcomes. The paired scientific experiment and full budget remain intact.

# Full matched sampling comparison, October 1

Both ten-epoch arms were submitted once: **866715 uniform control**, **866716
stratified candidate**. At the first queue check uniform was RUNNING on evc34
and stratified PENDING Priority. Allocation alone is not optimizer acceptance;
inspect the live header, initialization, run and terminal records. Never repeat
these dispatches or use their output directories for another attempt.

This experiment tests whether varying slice coverage across epochs helps find
focal evidence at a fixed observation budget, or instead dilutes useful evidence.
The only recipe difference is `slice_sampling`. Both start from the same pinned
official DINOv3-L weights and zero-initialized rank32 residual adapters, use the
same 2,247 training studies, 1,156 untouched outer studies, six selected series,
sixteen slices, 224-pixel input, head, optimizer, seed and 2,810-update budget.
Evaluation remains uniform. The primary comparison is epoch10 on the same outer
pool, with all twelve targets and leakage-component uncertainty; no epoch selection.
A positive result would still need another seed/fold before a broad conclusion.

## Actual prerequisites

- GPU pilot866624 completed0:0. All120 adapter modules changed,415 base tensors
  stayed unchanged; decoded final model/optimizer/RNG and MRI predictions match
  after the two-stage interruption/resume. Raw Torch ZIP hashes differ and are
  not claimed equal. Actual initial learned tensors and zero-adapter logits agree.
- That diagnostic is private at HF906732feec5fe3b67e61b94cec6ebce7d3dd4bec;
  all26 objects were fully downloaded/size/SHA checked. It is not an initialization
  checkpoint, competition candidate or accuracy result.
- CPU866694 completed0:0, verifying the full82,834,493,440-byte accepted256 cache
  archive and exact48,772-member/82,794,366,615-byte extraction. All3,403 studies/
  18,906 series metadata bindings were checked, with bounded actual pixel decode
  for32 studies. Original masks, labels, folds and source remain unchanged.
- Rebuildable972 `.pyc` files and one redundant1,215,686,736-byte upload ZIP were
  retired after exact validation, retaining every Python source and all43 bundle
  payloads. Private cloud data/checkpoints were not deleted. Initial admission
  reserves the two runs, existing native-cache obligations and a20GiB/5,000-file
  floor.22 checkpoint slots at128MiB plus16MiB/400-file metadata margin are funded.

## Immutable implementation and recovery

Scientific source is7ac7a9779d3c8bfd756a44eceb6414e9bb763b85, independently accepted
as45 materialized declared files. The existing9.23MB source pack, frozen encoder,
private environment and original cache are reused. Runtime cache/environment/code
are extracted onto owned node-local scratch and cleaned after the child exits.
Each role receives its own unique permanent output and private publication prefix.
Exact stage payloads and hashes are in `files.json`; one shared dispatch claim
rechecks actual accepted CPU/GPU receipts, asset signatures, quota and GPU count
immediately before each scheduler mutation. There are at most two one-H100/two-CPU
nodes. Failed-node exclusions include evc22; no automatic retry is allowed.

The runtime repeats actual full-config initialization and compares it to the
accepted pilot. Strict deterministic CUDA algorithms apply to both roles.
Scheduler USR1/TERM/INT requests go through the runtime to the original supervisor,
which checkpoints workers at optimizer boundaries. An actual process/signal test
verified the complete parent→supervisor→worker chain and fresh checkpoint receipt.
The H100 pilot separately tested actual optimizer recovery. Any production stop
requires exact terminal/checkpoint/recovery review before a distinct resume launch.

Every completed epoch has an immutable trainer snapshot. The dedicated
`publish_sampling_epoch.py` refuses diagnostic/incomplete snapshots, incorrect
recipes/roles/membership/source/init provenance or altered payloads; publication
is private, uses one CPU/zero GPUs and the shared atomic HF exclusion, and verifies
all five objects by full re-download/size/SHA. `dispatch_sampling_epoch.py` streams
only the ignored local credential through SSH stdin. Existing intent/journal/path
means reconcile read-only before any retry. No Kaggle submission was made.

The native384 source lanes and assembly remain intact. The full256 cache is ready
now, so useful paired observation training can run while the independent384
source cache finishes. The native dispatcher must respect these two actual GPU
roles and all remaining byte/file obligations. This changes scheduling, not the
scientific native384 experiment. No held-out or Kaggle gain is claimed yet.
