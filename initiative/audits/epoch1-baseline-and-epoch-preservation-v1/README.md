# First six full-data epochs and private preservation

The accepted report-supervision release, masks, folds, source cache and official
DINOv3-L weights are unchanged. The exact trainer source is Git
`b74060b98f01bff1b924070ccba5ce44dd002f40`. Primary Newton job **841107**
continued from the verified update-26 recovery point on evc28 with two H100s
and four training CPUs. It completed three full-data epochs on 2,247 eligible
training studies while this audit was captured, and remained running. The
trainer's fixed monitoring cadence made immutable epoch-2 sidecar/checkpoint
SHA `f56c0d64519fbf2fde5433ef4ff3dcc59d8f01394d80515ebf5cebf97726fa02`.

The owner requested a separate epoch-1 Kaggle baseline after its rolling
recovery file had already been replaced. Two-H100 replay job **841126** resumed
the *same* proven update-26 checkpoint and reached the completed epoch-1
boundary, step 281. The trainer intentionally exited 75 after saving its
checkpoint; Torchrun propagated the nonzero status to the wrapper, so Slurm
reported `FAILED 1:0` even though the saved state was complete. CPU job
**841157** verified its actual checkpoint, source and release bindings, stop
receipt, and all **255** resumed optimizer events against primary job 841107.
The replay checkpoint SHA is
`01d56458c54f4f2811ecfd5b997e99baec765e850345001960b953304ff3d28e`.
CPU job **841160** then hard-linked that unchanged checkpoint into an immutable
epoch-1 sidecar. This is an explicit one-off exception for Kaggle inference;
the trained two-epoch submission cadence remains unchanged.

CPU-only watcher **841134** shares physical node evc28, detects completed odd
epoch boundaries from the primary trainer log, hard-links the recovery inode
before its next replacement, then checks the checkpoint with Torch. Its first
capture, epoch 3 / step 843, passed with SHA
`caaa821440a1e880ba95744223ac693d30a5f8d3ad62a4d0a0f29d46e2b494f4`.
It must remain healthy to capture epochs 5, 7 and 9; a submitted watcher alone
does not prove those future snapshots. Even epochs 4, 6, 8 and 10 remain the
trainer's own scheduled snapshots.

All first six checkpoint files were committed to the **private**
[`GodlyDonuts/RSNA-Data`](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data)
dataset from CPU allocations on Newton. Each returned revision was followed by
a complete private download of the checkpoint, sidecar, official DINOv3
license, manifest and any baseline proof; their bytes and SHA-256 matched the
original cluster files. Large checkpoint bytes are held on Newton and private
Hugging Face, not in Git or on the laptop.

| Completed epoch | Checkpoint SHA-256 prefix | Verified private HF revision |
| --- | --- | --- |
| 1 | `01d56458c54f` | [`94d95aa2`](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/94d95aa232f41e2859dd56e5134901efd066a406/experiments/dinov3-v1/training-epochs/fold0-seed20260922/epoch-001) |
| 2 | `f56c0d64519f` | [`1ef2a32c`](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/1ef2a32c33fa635b5c9498e14eeb11afaf0e428d/experiments/dinov3-v1/training-epochs/fold0-seed20260922/epoch-002) |
| 3 | `caaa821440a1` | [`2d777a58`](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/2d777a58bf4a8a247c6be7b5943ac11ee40d67f3/experiments/dinov3-v1/training-epochs/fold0-seed20260922/epoch-003) |
| 4 | `90a72a4f203d` | [`fc78e3d4`](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/fc78e3d40b5a00a69bcab7be30d5cf732eb552de/experiments/dinov3-v1/training-epochs/fold0-seed20260922/epoch-004) |
| 5 | `8ea11a5b513e` | [`eac9a186`](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/eac9a18621048c395713eb81d40e6a0955120fc0/experiments/dinov3-v1/training-epochs/fold0-seed20260922/epoch-005) |
| 6 | `22f6badee0c5` | [`6d6cdeb6`](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/6d6cdeb6434114d7508a6d0bf8b992c68b72b90b/experiments/dinov3-v1/training-epochs/fold0-seed20260922/epoch-006) |

The Kaggle wheelhouse was staged and fully rehashed in CPU job **841116**.
The Kaggle management environment was installed in CPU job 841121, whose
import-only probe failed because the SDK authenticates on import. CPU recovery
job **841131** completed the exact pinned version/import checks with a
nonsecret offline placeholder; no real Kaggle credential was stored remotely.
The isolated offline HF client passed CPU job **841139**. Actual Hub tokens
were streamed from the local ignored secret store through SSH/Slurm stdin for
each upload, without persistent remote secret files. The selected original
dispatches, scripts and receipts are under `local-artifacts/`; exact small
cluster receipts, snapshot sidecars and HF publication journals are bound by
`source-files.json`.

GitHub main commit `2bc26959c4b7d3c82a7cba41f8b8f662970a91ac` includes the
one-off baseline adapter and epoch publisher; its pipeline checks passed at
[run 35815247444](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/actions/runs/35815247444).
The exact committed-code overlay passed full private HF byte verification at
revision `dabd9b0ea40f12f334f226b4a9163a0becd737ba`.

No MRI-only Kaggle notebook execution or competition submission has completed
at this audit point. The three-row public example CSV is not a leaderboard
baseline, and no score or hidden-test runtime is claimed.
