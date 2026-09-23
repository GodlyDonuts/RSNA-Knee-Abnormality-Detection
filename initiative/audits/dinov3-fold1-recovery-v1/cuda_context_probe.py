"""Require a real CUDA context and NCCL collective on both assigned ranks."""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 2 or rank not in (0, 1) or torch.cuda.device_count() != 2:
        raise RuntimeError("Exactly two scheduler-assigned CUDA ranks are required")
    torch.cuda.set_device(rank)
    value = torch.tensor([rank + 1], device=f"cuda:{rank}", dtype=torch.float32)
    torch.cuda.synchronize(rank)
    dist.init_process_group("nccl")
    dist.all_reduce(value)
    torch.cuda.synchronize(rank)
    if value.item() != 3:
        raise RuntimeError("Two-rank NCCL result differs")
    print(
        json.dumps(
            {
                "status": "allocated_cuda_context_and_nccl_passed",
                "job_id": os.environ["SLURM_JOB_ID"],
                "node": os.uname().nodename,
                "local_rank": rank,
                "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "collective_value": value.item(),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
