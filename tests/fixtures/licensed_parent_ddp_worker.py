"""Fresh two-process CPU engineering exercise; no MRI or accuracy evaluation."""

import os
import random
import runpy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from rsna.competitive.licensed_parent_train import Recipe, fit


def worker(rank, root, mode, test_path):
    torch.set_num_threads(1)
    random.seed(111 + rank)
    np.random.seed(111 + rank)
    torch.manual_seed(111 + rank)
    symbols = runpy.run_path(test_path)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(root) / (mode + ".rendezvous")),
        rank=rank,
        world_size=2,
    )
    try:
        data = symbols["Cases"]()
        model = symbols["TinyMRI"]()
        output = Path(root) / ("full" if mode == "full" else "resumed")
        fit(
            model,
            data,
            output,
            symbols["binding"](),
            Recipe(epochs=2),
            device=torch.device("cpu"),
            resume=mode == "resume",
            stop_after=1 if mode == "stop" else None,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""
    root, mode, test_path = sys.argv[1:]
    mp.spawn(worker, args=(root, mode, test_path), nprocs=2, join=True)
