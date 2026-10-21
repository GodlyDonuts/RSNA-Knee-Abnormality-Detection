"""Distinct all-data C100 seed20261021 on the owner's Colab A100.

Same accepted pixel/target bytes and original numerical training functions.
This is an ensemble-diversity candidate, not a new causal seed comparison or
independent clinical validation. No Kaggle submission is made here.
"""

import ast
import contextlib
import gc
import hashlib
import io
import json
import os
import random
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  HF_HUB_DISABLE_TELEMETRY='1', HF_HUB_DISABLE_PROGRESS_BARS='1')
ROOT = Path('/content/rsna-independent-C100-20261021-v1')
REPO = 'GodlyDonuts/RSNA-Colab-Deadline-20261021'
RUN = ROOT / 'run-C100-seed20261021'
PREFIX = 'runs/C100-seed20261021'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def once(path, value):
    with Path(path).open('xb') as f:
        f.write(canonical(value) + b'\n')


def main(token):
    os.umask(0o077)
    assert not RUN.exists(), 'Do not duplicate or overwrite an original run'
    inputs = json.loads((ROOT / 'input-verified.private.json').read_text())
    assert inputs['all_original_member_bytes_SHA_verified_on_Colab'] is True
    assert inputs['seed'] == 20261021 and inputs['cohort'] == 3887
    for name, proof in inputs['members'].items():
        if not name.startswith('contexts/'):
            assert sha(ROOT / name) == proof['sha256']
    sys.path.insert(0, str(ROOT / 'code'))
    import numpy as np
    import timm
    import torch
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
    from rsna.competitive.predictive.cache import signature
    from rsna.competitive.predictive.context_cache import load
    from rsna.competitive.predictive.model import Predictive
    from rsna.competitive.predictive.train import Recipe, fit
    assert torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0)
    assert torch.cuda.get_device_properties(0).total_memory >= 75 * 2**30
    assert torch.__version__.startswith('2.11.0') and timm.__version__ == '1.0.29'
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert shutil.disk_usage('/content').free > 40 * 2**30
    api = HfApi(token=token)
    assert api.repo_info(REPO, repo_type='dataset').private
    assert not list(api.get_paths_info(REPO, paths=[PREFIX], repo_type='dataset'))
    rows = inputs['rows']; uids = [r['uid'] for r in rows]
    assert hashlib.sha256(json.dumps(uids, separators=(',', ':')).encode()).hexdigest() == inputs['original_fitting_order_sha256']
    with np.load(ROOT / 'targets.private.npz', allow_pickle=False) as a:
        assert a['uids'].tolist() == uids
        labels = torch.from_numpy(a['labels'].copy()).float()
        masks = torch.from_numpy(a['masks'].copy()).bool()
    assert labels.shape == masks.shape == (3887, 12)
    assert torch.isfinite(labels[masks]).all() and ((labels[masks] >= 0) & (labels[masks] <= 1)).all()
    for row in rows:
        path = ROOT / row['file']
        row['proof'] = {**row['original_proof'], 'stat': signature(path)}
    selected = [n for n in ast.parse((ROOT / 'parent_source.private.py').read_text()).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                and n.name in ('build_backbone', 'RaptorClassifier', 'load_model')]
    assert len(selected) == 3
    ns = dict(torch=torch, nn=torch.nn, timm=timm, gc=gc, hashlib=hashlib, INPUT_RES=320)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'original_parent_encoder', 'exec'), ns)
    with contextlib.redirect_stdout(io.StringIO()):
        parent, side = ns['load_model'](str(ROOT / 'parent.pt'),
            'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, torch.device('cpu'))
    assert side == 320 and sum(p.numel() for p in parent.parameters()) == 73136096

    class Encoder(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__(); self.backbone = backbone
            self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
            self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

        def forward(self, images):
            return self.backbone((images - self.mean) / self.std)

    recipe = Recipe(**inputs['recipe'])
    random.seed(recipe.seed); np.random.seed(recipe.seed)
    torch.manual_seed(recipe.seed); torch.cuda.manual_seed_all(recipe.seed)
    model = Predictive(Encoder(parent.backbone), parent.backbone.num_features, hidden=128)
    del parent
    assert sum(p.numel() for p in model.parameters()) == 73271124
    assert all(p.requires_grad for p in model.parameters())
    initial = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        initial.update(canonical([name, str(tensor.dtype), list(tensor.shape)]))
        initial.update(tensor.contiguous().numpy().tobytes())
    binding = {'branch': 'Colab-C100-seed20261021', 'source_git': inputs['source_git'],
        'base_science_git': inputs['base_science_git'], 'original_binding_sha256': inputs['original_binding_sha256'],
        'input_manifest_sha256': inputs['downloaded_manifest_sha256'],
        'input_private_revision': inputs['actual_private_revision'],
        'fitting_order_sha256': inputs['original_fitting_order_sha256'],
        'initial_complete_state_sha256': initial.hexdigest(), 'recipe': asdict(recipe),
        'parent_weight_sha256': inputs['members']['parent.pt']['sha256'],
        'parent_source_sha256': inputs['members']['parent_source.private.py']['sha256'],
        'targets_sha256': inputs['members']['targets.private.npz']['sha256'],
        'license_sha256': inputs['members']['source-permission.txt']['sha256'],
        'numerical_code_sha256': {n: p['sha256'] for n, p in inputs['members'].items() if n.startswith('code/')},
        'torch': torch.__version__, 'timm': timm.__version__, 'cuda': torch.version.cuda,
        'GPU': torch.cuda.get_device_name(0), 'new_seed_and_runtime_not_strict_causal_comparison': True,
        'protected_labels_or_external_source_data_used': False, 'clinical_accuracy_claim': False,
        'started_UTC': datetime.now(timezone.utc).isoformat()}
    RUN.mkdir(mode=0o700); once(RUN / 'run-start.json', binding)
    model.cuda(); gc.collect()
    started = time.monotonic()
    absolute_stop = datetime(2026, 10, 22, 4, tzinfo=timezone.utc).timestamp()
    pool = ThreadPoolExecutor(max_workers=4)

    class Cases:
        def __init__(self):
            self.epoch = None; self.order = []; self.cursor = 0; self.futures = {}

        def __len__(self):
            return len(rows)

        def read(self, index):
            row = rows[index]
            contexts = load(ROOT / row['file'], row['proof'], provenance=row['provenance'])
            return {'contexts': contexts, 'labels': labels[index:index+1], 'mask': masks[index:index+1]}

        def case(self, index, epoch):
            if epoch != self.epoch:
                assert not self.futures
                self.epoch = epoch; self.cursor = 0
                self.order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(recipe.seed + epoch)).tolist()
            assert self.order[self.cursor] == index
            for j in self.order[self.cursor:self.cursor+4]:
                if j not in self.futures:
                    self.futures[j] = pool.submit(self.read, j)
            result = self.futures.pop(index).result(); self.cursor += 1
            return result

    publication_pool = ThreadPoolExecutor(max_workers=1)
    publications = {}

    def preserve(epoch, checkpoint):
        prefix = f'{PREFIX}/epoch-{epoch:03d}'
        destination = RUN / f'publication-{epoch:03d}'; destination.mkdir(mode=0o700)
        assert api.repo_info(REPO, repo_type='dataset').private
        assert not list(api.get_paths_info(REPO, paths=[prefix], repo_type='dataset'))
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['step'] == epoch * 486 and state['binding'] == binding
        export = destination / 'model.pt'
        with export.open('xb') as f:
            torch.save({'model': state['model'], 'architecture': {'arm':'C', 'branch':binding['branch'],
                'encoder':'CoAtNet_RMLP2', 'resolution':320, 'hidden':128, 'classes':12,
                'descriptor_width':64, 'context_cap':24, 'halo':8},
                'provenance': {**binding, 'epoch':epoch, 'completed_updates':state['step']}}, f)
        del state
        files = {'model.pt': export, 'full-state.private.pt': checkpoint,
                 'run-start.json': RUN / 'run-start.json', 'source-permission.txt': ROOT / 'source-permission.txt',
                 'colab_train.py': ROOT / 'colab_train.py'}
        proofs = {n: {'bytes':p.stat().st_size, 'sha256':sha(p)} for n,p in files.items()}
        once(destination / 'upload-intent.json', {'prefix':prefix, 'files':proofs})
        commit = api.create_commit(REPO, repo_type='dataset', operations=[
            CommitOperationAdd(path_in_repo=f'{prefix}/{n}', path_or_fileobj=str(p)) for n,p in files.items()],
            commit_message=f'Preserve independent C100 seed20261021 epoch{epoch}')
        with tempfile.TemporaryDirectory(prefix='rsna-colab-verify-') as tmp:
            for name, proof in proofs.items():
                p = Path(hf_hub_download(REPO, f'{prefix}/{name}', repo_type='dataset',
                                         revision=commit.oid, token=token, cache_dir=tmp))
                assert p.stat().st_size == proof['bytes'] and sha(p) == proof['sha256']
        result = {'status':'private_epoch_full_download_size_SHA_verified', 'epoch':epoch,
                  'repo':REPO, 'revision':commit.oid, 'prefix':prefix, 'files':proofs}
        once(destination / 'verified.json', result)
        print(json.dumps({'phase':'private_epoch_preserved', 'epoch':epoch, 'revision':commit.oid}), flush=True)
        return result

    def seal():
        for epoch in range(1, 7):
            checkpoint = RUN / 'checkpoints' / f'epoch-{epoch:03d}.pt'
            if epoch not in publications and checkpoint.exists():
                publications[epoch] = publication_pool.submit(preserve, epoch, checkpoint)

    def update(row):
        assert time.monotonic() - started < 10 * 3600 and time.time() < absolute_stop
        assert shutil.disk_usage('/content').free > 20 * 2**30
        for f in publications.values():
            if f.done(): f.result()
        seal()
        value = {**row, 'elapsed_seconds':time.monotonic()-started,
                 'peak_GPU_bytes':torch.cuda.max_memory_allocated()}
        with (RUN / 'updates.jsonl').open('a') as f: f.write(json.dumps(value)+'\n')
        if row['step'] <= 4 or row['step'] % 20 == 0:
            print(json.dumps({'phase':'real_optimizer_progress', **value}), flush=True)

    print(json.dumps({'phase':'actual_full_model_training_start', 'studies':3887,
                      'updates':2916, 'seed':recipe.seed, 'parameter_count':73271124,
                      'source_git':inputs['source_git'], 'GPU':binding['GPU']}), flush=True)
    state = fit(model, Cases(), recipe, binding=binding, output=RUN/'checkpoints',
                on_update=update, epoch_snapshots=True)
    assert state['step'] == 2916 and state['cohort_size'] == 3887
    seal()
    assert set(publications) == set(range(1,7))
    for f in publications.values(): f.result()
    pool.shutdown(); publication_pool.shutdown()
    terminal = {'status':'actual_full_C100_seed20261021_complete', 'binding':binding,
                'updates':2916, 'epochs':6, 'elapsed_seconds':time.monotonic()-started,
                'all_epochs_private_verified':True, 'clinical_or_Kaggle_score':None}
    once(RUN/'terminal.json', terminal)
    api.create_commit(REPO, repo_type='dataset', operations=[CommitOperationAdd(
        path_in_repo=PREFIX+'/terminal.json', path_or_fileobj=str(RUN/'terminal.json'))],
        commit_message='Complete original independent Colab C100 full run')
    print(json.dumps(terminal), flush=True)


if __name__ == '__main__':
    main(json.load(sys.stdin)['token'])
