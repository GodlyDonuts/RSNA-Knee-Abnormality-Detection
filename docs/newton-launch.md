# Newton launch kit

These tools prepare and inspect a launch locally. Preparation does not log into Newton,
submit jobs, install packages, fetch large weights, or download the image collection.
The separate weight stager downloads only when explicitly requested inside a Slurm allocation.
They target `python -m rsna.competitive.train` and the accepted report-supervision
release. GPU training and preemption behavior still require an allocated-node pilot.

## Access and storage first

Confirm the actual account, GPU partition, available GPU resource names, walltime,
preemption policy, allocation balance, shared project storage, scratch policy and
quota with ARCC. `myusage`, `sinfo`, and `scontrol show partition` are read-only
starting points after access. The generator deliberately supplies no guessed
account, partition, memory capacity, GPU type or allocation. CPU/RAM/GPU requests
are explicit; GPU model selection is optional and must match the local GRES name.
Newton has heterogeneous hardware, and an allocation is shared rather than a
promise of dedicated H100 time. [UCF scheduler](https://arcc.ist.ucf.edu/docs/scheduler/),
[Newton resources](https://arcc.ist.ucf.edu/index.php/resources/newton/about-newton)

Keep sources, accepted manifests, logs and checkpoints in approved durable project
storage. Use node-local storage only for reproducible caches when site policy
allows it. Confirm free capacity and quota before transfers; filesystem free space
alone does not establish an account's quota. The accepted cache is about 82.6 GB,
and the original image collection about 569.8 GB. Stage immutable archives/shards
on cluster storage and use CPU allocations for validation, unpacking and decoding.
Do not transfer millions of individual DICOMs through the laptop or run heavy
preprocessing on login nodes. Use ARCC's approved transfer service and retained
hash manifests. [UCF storage policy](https://arcc.ist.ucf.edu/docs/data/files/)

## Reproducible environment and offline files

Create a **new** environment in approved project storage after determining the
compute-node driver/CUDA combination. Refuse an already existing environment path;
do not upgrade a shared environment in place. A Linux/CUDA environment cannot be
copied from this macOS laptop. Choose the compatible CUDA PyTorch wheel explicitly,
then install this repository's exact dependency versions in that environment.
The production encoders additionally require `transformers==5.6.1` and
`safetensors==0.7.0`; no `timm` or `torchvision` dependency is needed by these adapters.

For an offline deployment, resolve and download a platform-compatible wheelhouse
once on an authorized cluster transfer/build host, record each wheel's SHA-256,
and create a pinned requirements lock. Install into the new environment with
`--no-index --find-links` and `--require-hashes`. Resolve on the same Linux/Python/
CUDA target, not from laptop package metadata. Keep the lock, wheel hashes, Python
version and selected CUDA index URL with the run; exclude credentials and signed
URLs. Environment packing is optional after a successful compute-node smoke check;
an unpacked archive still needs path repair and a fresh preflight.

Fetch eligible model files only after the required access and license checks,
directly to appropriate project storage. Each training config's `encoder` binds
`architecture`, `config_path`, `config_sha256`, `weights_path`, `weights_sha256`,
`transformers_version`, `state_dict_format`, `source_id`, immutable 40-character
`source_revision`, and `license`. Paths are relative to the repository working
directory or absolute cluster paths. Never place tokens in this JSON. DINOv3 access
is conditional; no gate acceptance or entitlement is supplied by this kit.
`diagnostic_tiny` is permitted only with a positive `diagnostic_study_limit` and
does not stand in for a pretrained encoder test.

Use a clean clone at the exact code commit selected for the experiment. Keep
tracked configuration files unchanged; keep generated configs and outputs outside
the clone. The pinned weight paths use the ignored `artifacts/encoder-weights/`
directory inside the clone, which should itself reside on approved project storage.
The read-only preflight rejects uncommitted/untracked checkout changes. Freeze the fold overlay before
model selection and preserve release masks, exclusions, groups and evaluation-only
gold. No environment/bootstrap step changes the accepted release.

### Pinned public encoders

The two small preparation manifests are
`configs/competitive/weights/dinov2-registers-base.json` and
`configs/competitive/weights/convnext-tiny-224.json`. Copy the selected manifest's
`encoder` object into the training configuration. Their exact upstream config
bytes are archived beside them; weights are absent from Git.

| Encoder | Immutable revision | Weight bytes | Format |
| --- | --- | ---: | --- |
| `facebook/dinov2-with-registers-base` | `a1d738ccfa7ae170945f210395d99dde8adb1805` | 346,358,296 | safetensors |
| `facebook/convnext-tiny-224` | `6166b7613034066690a621d8bf25ffdf181a34f0` | 114,423,189 | PyTorch state dictionary |

Both pinned Hub metadata records report Apache-2.0 and public access. These facts
alone do not determine competition eligibility. Their weight hashes come from
upstream LFS metadata; staging verifies the full downloaded bytes before publication.
The ConvNext file includes its ImageNet classification head: the factory validates
that complete state before extracting the backbone.

From the repository directory, offline verification is the default:

```bash
python scripts/newton/stage_weights.py --repo "$PWD" \
  --manifest configs/competitive/weights/dinov2-registers-base.json
```

For a later authorized cluster transfer inside a CPU Slurm allocation, add
`--download`. This opt-in is not a scheduler submission. The stager streams only
the pinned official revision, verifies size and SHA-256, then publishes atomically
without replacing any existing destination. Existing incorrect files fail closed.
An interrupted transfer lock requires inspection; it is never removed as an
automatic retry. There is no access-token argument, gate acceptance, floating
revision, or arbitrary URL option. A verified result prints hashes and sizes,
not workspace paths or signed transfer URLs. Use the ConvNext manifest analogously.

## Generate and review a job

Run `python scripts/newton/launch.py --help`. Required options include:

- Scheduler: `--account`, `--partition`, `--time` (`HH:MM:SS` or `D-HH:MM:SS`),
  `--job-name`, `--cpus`, `--memory-gb`, `--min-free-gb`.
- Cluster paths: `--repo`, `--python` (the new environment's Python), `--studies`,
  `--series`, `--cache`, `--cache-manifest` (the accepted cache-index artifact),
  `--folds`, `--artifact-root`, `--logs`, `--config`, `--output`.
- Provenance: `--expected-git` is the full 40-character code commit, never `main`.
- Optional: `--gpus 1|2`, `--gpu-type`, `--mode independent|ddp`, `--resume`,
  `--signal-seconds`, `--checkpoint-grace-seconds`, `--write-script`.

Every path must be concrete and absolute; unresolved shell expansions and example
placeholders are rejected. Paths with spaces are quoted safely. No path on Newton
needs to exist on the laptop merely to generate the plan. The default is one GPU.
With two GPUs, **independent** mode is still the default: supply two `--config`
and two distinct `--output` paths for two separate fold/seed experiments, sharing
the node's requested CPU/RAM budget. Each config should have a compatible worker
count. `--mode ddp --gpus 2` instead launches one configuration with two ranks.
Use one-GPU jobs for better scheduler flexibility when pairing is unnecessary.

The output is JSON containing the script, its hash and two **unevaluated** commands.
`--write-script` saves a new file exclusively; it never overwrites one. There is no
submission mode, even without `--dry-run`. On Newton, create the log directory
**before** manually invoking `sbatch`: Slurm opens stdout/stderr before the script
starts, so an in-script `mkdir` would be too late. Review the generated resource
requests and script before any submission. The script uses one node and one Slurm
task; `torchrun` starts one or two trainer ranks inside the assigned GPU mask.
[UCF submission examples](https://arcc.ist.ucf.edu/docs/scheduler/scripts/)

## Preflight and graceful interruption

The script runs `scripts/newton/preflight.py` before training. It records actual
CPU/RAM, CUDA visibility, GPU memory, driver, PyTorch CUDA build, NCCL version,
GPU/CPU topology, free storage, package versions, code commit and local cache/model
artifact hashes. It rejects a CUDA request without the exact requested visible GPU
count; it never switches to CPU. Two-rank DDP requires NCCL, and BF16 requires a
positive device capability check. The report omits full paths, environment dumps,
config bodies, tokens and subprocess error payloads. Its hashes identify local
artifacts; they do not independently certify every cache array. The trainer retains
responsibility for full release/cache/fold verification.

Slurm's `B:USR1` warning reaches the batch shell. Its trap forwards USR1 to each
supervisor, which signals **trainer workers**, not the elastic agent or data-loader
descendants. The trainer coordinates ranks, completes the current effective update,
saves `output/checkpoint.pt` atomically, and exits 75. The supervisor requires a
fresh trainer stop receipt with exit code 75, a binding hash matching `run.json`,
and the exact checkpoint SHA-256; a periodic checkpoint alone is insufficient.
It then returns 75 even when torchrun reports a generic worker failure.
It terminates a stuck process group after the grace period and reports failure if
no valid stop receipt exists. Final-inference interruption may point to the already
saved final checkpoint through the same exact hash. TERM/INT request the same checkpoint path as a best
effort; scheduler hard-kill timing may leave less time. No automatic requeue occurs.

Default warning lead is 300 seconds and grace is 120 seconds. The generator requires
at least 120 seconds between them; pilot actual checkpoint latency and increase both
for a larger model. Slurm may issue the warning up to 60 seconds early, and its
walltime TERM-to-KILL interval is site-specific. Resume explicitly from
`output/checkpoint.pt` with the same world size and compatible input/config hashes.
Retain failed/preempted logs; do not silently restart as a new seed.
[Slurm signal and walltime behavior](https://slurm.schedmd.com/sbatch.html)

Before a larger campaign, perform one bounded CUDA pilot per encoder, a two-rank
NCCL/gradient-equivalence pilot if using DDP, and a real Slurm USR1 checkpoint/resume
rehearsal. Offline shell/signal tests do not establish Newton resource availability,
GPU memory fit, scheduler policy or exact GPU numerical resume.


## Approved DINOv3-L and node-local archives

The approved `configs/competitive/weights/dinov3-large.json` pins the actual
upstream config, 1,212,559,808-byte safetensors file, and complete DINOv3 license.
This model uses the custom DINOv3 agreement (`license: other`), not Apache-2.0.
Keep the full license with redistributed model derivatives. Competition
eligibility is a separate check. Authenticated downloads read an existing
`HF_TOKEN` environment value, never a command-line token; redirects remove it
before leaving the exact Hub host. The stager does not accept access terms.

`materialize_config.py` combines `configs/competitive/dinov3-campaign.json` with
that exact manifest only after offline verification of the staged config,
weights, and license. Specify `--recipe`, `--manifest`, `--repo` (the encoder
staging root), `--output`, and `--mode pilot|production`. Outputs bind absolute
cluster encoder paths so weights can remain separate from a clean code clone.
An identical existing config is accepted; a different one is preserved and
rejected. Pilot configurations have 100 studies, two epochs, and no submission
cadence. Production follows the fixed ten-epoch recipe; initial fold is zero.

Newton's measured Lustre inode headroom cannot hold the 48,772-file full cache.
`stream_cache_archive.py` streams a deterministic archive directly through a
private SSH ControlMaster, without a laptop archive. It retains an exclusive
partial on failure, verifies the received archive bytes and member count, then
publishes without replacing any existing target. An uncertain transfer requires
read-only reconciliation, not an automatic repeat. Its receipt contains the
final SHA, archive bytes, file count and unpacked bytes required below.

For archive mode, replace `--cache` with `--cache-archive`,
`--cache-archive-sha256`, `--cache-archive-files`, `--cache-unpacked-bytes` and
`--scratch-base /tmp`. The generator requires exact concrete values. On the
allocated node, `runtime_archive.py` checks real local disk and at least 100 GiB
free (plus the requested reserve when the cache is larger), creates a private
job-owned directory, hashes the exact stream being unpacked, and rejects links,
unsafe names, mismatched bytes and inventory. The launch cleans that owned tree
on exit; TERM/INT while unpacking also clean it. Runtime capacity is checked on
every node: the measured 716 GiB free on evc22 is not a guarantee for other nodes.

Keep the environment as a separate verified archive on Lustre. Supply
`--environment-archive`, `--environment-archive-sha256`,
`--environment-archive-files` (regular members only), and
`--environment-unpacked-bytes` alongside cache archive mode. In this mode,
`--python` is the verified shared base executable (`/usr/bin/python3.12` on the
measured cluster). The launcher uses it to extract a unique private node-local
environment, then switches to that environment's actual interpreter for all
preflight, supervisor and trainer calls. Cleanup removes both owned roots.
The environment helper verifies the exact stream SHA and inventory, rejects
unsafe names/hard links, writes files before installing contained relative links,
and permits external Python aliases only to that exact shared executable.
Retain the builder's relocation acceptance and full dependency lock. No packages
are installed by this launch. Do not run archived pip/pytest entry-point scripts
whose shebangs still name the build directory.

For the first bounded recovery pilot, use `--acceptance --mode ddp --gpus 2`, one
pilot config and one fresh output directory. The launcher calls
`local_competitive_acceptance.py` with the actual node-local cache and interpreter,
CUDA, two ranks, and public examples disabled. That helper runs continuous
training, a deliberate interrupted update, and resume, then checks their results.
Batch USR1/TERM/INT is forwarded; the helper terminates and reaps its owned child
process group before failing, and the launch then cleans both extracted roots.
An interrupted diagnostic pilot produces no accepted completion receipt and may
be rerun in a fresh output directory. Production uses the checkpoint-verifying
supervisor instead. The initial explicitly selected resource plan is account
`skattel`, partition `normal`, one node, two `nvidia_h100_pcie` GPUs, four CPUs,
32 GiB host RAM, and 30 minutes, subject to actual pilot measurements. No launch
is submitted by the generator. It uses a localhost rendezvous endpoint for
single-node production torchrun.
