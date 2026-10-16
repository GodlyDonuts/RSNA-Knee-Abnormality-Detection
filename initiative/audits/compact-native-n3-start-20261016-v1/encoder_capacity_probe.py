"""Bounded architecture capacity diagnostic; no clinical data, checkpoint or accuracy."""
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

R = Path('/tmp/rsna-compact-n3-capacity-20261016-v1')
assert not R.exists()
assert os.uname().nodename == 'evc40'
assert shutil.disk_usage('/tmp').free > 396100843459 + 20 * 2**30
R.mkdir(mode=0o700)
os.environ['CPATH'] = '/tmp/rsna-python312-devel-20261016-v1/usr/include/python3.12'
os.environ['TRITON_CACHE_DIR'] = str(R / 'triton')
os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(R / 'inductor')
import torch
import timm
from safetensors.torch import load_file
from torch import nn

assert timm.__version__ == '1.0.30'
assert torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name(0)
assert torch.cuda.is_bf16_supported()
random.seed(20260922)
torch.manual_seed(20260922)
torch.cuda.manual_seed_all(20260922)
torch.set_num_threads(2)
assets = Path('/tmp/rsna-compact-native-n3-assets-20261016-v1')
proof = json.loads((assets / 'stage-verified.json').read_text())
p = assets / 'initialization/model.safetensors'
assert p.stat().st_size == proof['files']['model.safetensors']['bytes']
# The original node staging proof already hashes these exact immutable bytes.
encoder = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=False,
                            drop_rate=0., drop_path_rate=0.)
encoder.load_state_dict(load_file(str(p)), strict=True)
encoder.reset_classifier(0)
for module in encoder.modules():
    if isinstance(module, nn.Dropout):
        module.p = 0.
assert sum(p.numel() for p in encoder.parameters()) == 72856264
assert all(p.requires_grad for p in encoder.parameters())
assert sum(isinstance(m, nn.modules.batchnorm._BatchNorm) for m in encoder.modules()) == 25
class Bag(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = encoder
        self.norm = nn.LayerNorm(1024)
        self.attention = nn.Sequential(nn.Linear(1024, 256), nn.Tanh(), nn.Linear(256, 12))
        self.dropout = nn.Dropout(.2)
        self.classifiers = nn.ModuleList([nn.Linear(1024, 1) for _ in range(12)])
    def forward(self, x):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            z = self.encoder(x)
        h = self.norm(z.float())
        a = self.attention(h).softmax(dim=0)
        pooled = self.dropout(a.T @ h)
        return torch.cat([head(pooled[i]) for i, head in enumerate(self.classifiers)])
model = Bag().cuda().train()
head = [p for name, p in model.named_parameters() if not name.startswith('encoder.')]
optimizer = torch.optim.AdamW([{'params': model.encoder.parameters(), 'lr': 3e-5},
                              {'params': head, 'lr': 1e-3}], weight_decay=.02)
bn = next(m for m in encoder.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm))
rows = []
for windows in [12, 1]:
    optimizer.zero_grad(set_to_none=True)
    x = torch.rand(windows, 3, 384, 384, device='cuda') * 2 - 1
    counter = int(bn.num_batches_tracked)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    logits = model(x)
    loss = nn.functional.binary_cross_entropy_with_logits(logits, torch.full_like(logits, .5))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(p.grad is not None for p in model.parameters())
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3., error_if_nonfinite=True)
    before = next(model.encoder.parameters()).detach().clone()
    optimizer.step()
    assert not torch.equal(before, next(model.encoder.parameters()).detach())
    assert int(bn.num_batches_tracked) == counter + 1
    torch.cuda.synchronize()
    rows.append({'windows': windows, 'seconds': time.monotonic() - start,
                 'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'loss': float(loss.detach()),
                 'gradient_norm': float(norm), 'gradient_tensors': len(gradients),
                 'BN_updates': 1, 'trunk_changed': True})
    del x, logits, loss, before
model.eval()
with torch.inference_mode():
    x = torch.zeros(1, 3, 384, 384, device='cuda')
    a = model(x)
    b = model(x)
    assert torch.equal(a, b) and torch.isfinite(a).all()
result = {'status': 'synthetic_capacity_forward_backward_passed', 'job_id': os.environ['SLURM_JOB_ID'],
          'GPU': torch.cuda.get_device_name(0), 'source_weight_sha256': proof['files']['model.safetensors']['sha256'],
          'timm': timm.__version__, 'trunk_parameters': 72856264, 'cases': rows,
          'clinical_data_read': False, 'checkpoint_saved': False, 'accuracy_claim': False,
          'integrated_trainer_recovery_accepted': False, 'full_training_dispatched': False}
(R / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
