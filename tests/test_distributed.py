from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from rsna.model import masked_bce


def check_rank(rank: int, rendezvous: str) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(5)
    inputs = torch.randn(4, 3)
    labels = torch.randint(2, (4, 12)).float()
    mask = torch.zeros_like(labels)
    mask[:2, :6] = 1
    mask[2, :9] = 1
    mask[3, :11] = 1
    reference = nn.Linear(3, 12)
    masked_bce(reference(inputs), labels, mask).backward()
    dist.init_process_group("gloo", init_method=rendezvous, world_size=2, rank=rank)
    try:
        model = nn.Linear(3, 12)
        model.load_state_dict(reference.state_dict())
        wrapped = DistributedDataParallel(model)
        section = slice(rank * 2, rank * 2 + 2)
        masked_bce(wrapped(inputs[section]), labels[section], mask[section]).backward()
        torch.testing.assert_close(model.weight.grad, reference.weight.grad)
        torch.testing.assert_close(model.bias.grad, reference.bias.grad)
    finally:
        dist.destroy_process_group()


def test_distributed_masked_gradients_equal_global_batch(tmp_path: Path) -> None:
    mp.spawn(check_rank, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
