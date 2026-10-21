"""Fixed-final C100 readout decomposition on label-blind selected fitting cases.

This is a failure investigation, NOT independent validation or a new model.
Original inputs and weights are read only. Per-case output stays on evc27.
"""
import ast
import contextlib
import gc
import hashlib
import io
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path('/tmp/rsna-predictive-readout-diagnostic-20261021-v1')
CONTROL = Path('/tmp/rsna-predictive-density100-full-20261020-v1')
RUN = Path('/tmp/rsna-predictive-full-C100-20261020-v1')
BINDING = 'f0d73e6cc04cbf7b34f06499f5aef4ba60ea2c593662764b8f64ec5105a289cd'
MODEL = 'e606d974f4aa8bcc2a4e07375353e41f1e4d31d7151f7d223380624c8116d8d6'
ROSTER = '0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42'
SELECTION = 'C100-fixed-final-readout-diagnostic-20261021'


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def once(p, v):
    with Path(p).open('x') as f:
        json.dump(v, f, sort_keys=True, allow_nan=False)
        f.write('\n')


def select(uids, n=128):
    assert len(uids) == len(set(uids)) and len(uids) >= n
    return sorted(range(len(uids)), key=lambda i: hashlib.sha256(
        (SELECTION + ':' + uids[i]).encode()).digest())[:n]


def summarize(q, mask, prior, full, bf):
    import numpy as np
    assert q.shape == mask.shape == prior.shape == full.shape == bf.shape
    assert q.ndim == 2 and q.shape[1] == 12 and mask.dtype == bool
    assert np.isfinite(q[mask]).all() and ((q[mask] >= 0) & (q[mask] <= 1)).all()
    assert np.isfinite(prior).all() and np.isfinite(full).all() and np.isfinite(bf).all()
    assert ((prior >= 0) & (prior <= 1) & (full >= 0) & (full <= 1)).all()
    def bce(p):
        p = np.clip(p.astype('float64'), 1e-7, 1 - 1e-7)
        # Do not let ignored NaN labels contaminate masked summaries.
        y = np.where(mask, q, 0)
        return -(y * np.log(p) + (1 - y) * np.log1p(-p))
    lp, lf = bce(prior), bce(full)
    result = []
    for j in range(12):
        m = mask[:, j]; count = int(m.sum())
        result.append(dict(target_index=j, observed_labels=count,
            context_BCE=float(lp[m, j].mean()) if count else None,
            full_BCE=float(lf[m, j].mean()) if count else None,
            full_minus_context_BCE=float((lf[m, j]-lp[m, j]).mean()) if count else None,
            context_probability_mean=float(prior[:, j].mean()),
            context_probability_std=float(prior[:, j].std()),
            full_probability_mean=float(full[:, j].mean()),
            full_probability_std=float(full[:, j].std()),
            probability_shift_mean_abs=float(np.abs(full[:, j]-prior[:, j]).mean()),
            logBF_RMS=float(np.sqrt(np.mean(bf[:, j]**2)))))
    assert mask.any()
    return dict(studies=len(q), observed_labels=int(mask.sum()), targets=result,
        context_BCE=float(lp[mask].mean()), full_BCE=float(lf[mask].mean()),
        full_minus_context_BCE=float((lf-lp)[mask].mean()),
        probability_shift_mean_abs=float(np.abs(full-prior).mean()),
        fitting_only=True, independent_accuracy_claim=False)


def main():
    os.umask(0o077)
    assert os.uname().nodename == 'evc27'
    assert not (ROOT/'started.json').exists(), 'Reconcile original diagnostic; do not repeat'
    funding = json.loads((ROOT/'admission.json').read_text())
    assert funding['new_node_cap_bytes'] == 16*2**20
    assert shutil.disk_usage('/tmp').free > (20+518)*2**30 + (176+16)*2**20
    assert sha(CONTROL/'binding.private.json') == BINDING
    binding = json.loads((CONTROL/'binding.private.json').read_text())
    terminal = json.loads((RUN/'terminal.json').read_text())
    assert terminal['updates'] == 2916 and terminal['epochs'] == 6
    assert terminal['binding_sha256'] == BINDING and terminal['GPU_job'] == '995351'
    code = Path(binding['code_root'])
    for name, h in binding['code_sha256'].items():
        assert sha(code/name) == h
    refs = {}
    for name in ('context_index', 'targets', 'parent_source'):
        ref = binding[name]; p = Path(ref['path'])
        assert sha(p) == ref['sha256']
        refs[name] = p
    rows = json.loads(refs['context_index'].read_text())
    uids = [r['uid'] for r in rows]
    assert len(uids) == 3887 and uids == sorted(set(uids))
    assert hashlib.sha256(json.dumps(uids, separators=(',', ':')).encode()).hexdigest() == ROSTER
    indices = select(uids)
    model_path = RUN/'publication/epoch-006/model.pt'
    # New exact readout uses the same fully preserved competition checkpoint.
    assert model_path.stat().st_size == 293424153 and sha(model_path) == MODEL
    sys.path[:0] = [str(code), *binding['python_paths']]
    import numpy as np
    import timm
    import torch
    from rsna.competitive.predictive.context_cache import load
    from rsna.competitive.predictive.density import Supports
    from rsna.competitive.predictive.model import Predictive
    from rsna.competitive.predictive.cache import signature
    assert torch.cuda.device_count() == 1 and 'H100' in torch.cuda.get_device_name(0)
    assert dict(torch=torch.__version__, timm=timm.__version__, cuda=torch.version.cuda) == binding['runtime_versions']
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    assert signature(binding['parent_weight']['path']) == binding['parent_weight']['stat']
    selected = [n for n in ast.parse(refs['parent_source'].read_text()).body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ('build_backbone', 'RaptorClassifier', 'load_model')]
    assert len(selected) == 3
    ns = dict(torch=torch, nn=torch.nn, timm=timm, gc=gc, hashlib=hashlib, INPUT_RES=320)
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'original_parent_constructor', 'exec'), ns)
    with contextlib.redirect_stdout(io.StringIO()):
        parent, side = ns['load_model'](binding['parent_weight']['path'],
            'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', 384, torch.device('cpu'))
    assert side == 320
    class Encoder(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__(); self.backbone = backbone
            self.register_buffer('mean', torch.tensor([.485,.456,.406])[None,:,None,None])
            self.register_buffer('std', torch.tensor([.229,.224,.225])[None,:,None,None])
        def forward(self, images):
            return self.backbone((images-self.mean)/self.std)
    model = Predictive(Encoder(parent.backbone), parent.backbone.num_features, hidden=128)
    del parent
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    assert state['provenance']['completed_updates'] == 2916 and state['provenance']['epoch'] == 6
    assert state['provenance']['binding_sha256'] == BINDING
    model.load_state_dict(state['model'], strict=True)
    del state
    model.cuda().eval(); gc.collect()
    with np.load(refs['targets'], allow_pickle=False) as a:
        assert a['uids'].tolist() == uids
        labels = a['labels'][indices].copy(); masks = a['masks'][indices].copy().astype(bool)
    started = time.monotonic()
    receipt = dict(status='fixed_final_fitting_readout_started', GPU_job=os.environ['SLURM_JOB_ID'],
        original_GPU_job='995351', original_model_sha256=MODEL, binding_sha256=BINDING,
        diagnostic_sha256=sha(__file__), source_git=funding['source_git'], studies=128,
        selection_rule=SELECTION, selection_label_blind=True,
        selected_roster_sha256=hashlib.sha256(json.dumps([uids[i] for i in indices], separators=(',', ':')).encode()).hexdigest(),
        fitting_roster_sha256=ROSTER, precision='FP16', training_or_checkpoint_mutation=False,
        protected_or_confirmation_cases=False, independent_accuracy_claim=False)
    once(ROOT/'started.json', receipt)
    priors, fulls, bfs = [], [], []
    with torch.inference_mode():
        for position, index in enumerate(indices):
            assert time.monotonic()-started < 1100, 'Bounded diagnostic expired'
            r = rows[index]; contexts = load(r['path'], r['proof'], provenance=r['provenance'])
            p, f, bf = [], [], []
            for c in contexts:
                args = [c[k].cuda() for k in ('images','token_metadata','region_metadata','targets','observed')]
                s = c['supports']; s = Supports(*(x.cuda() for x in (s.indptr,s.regions,s.log_prior)),s.region_count)
                with torch.autocast('cuda', dtype=torch.float16):
                    out = model(*args,s)
                p.append(out['context_logits'].float().sigmoid().cpu())
                f.append(out['logits'].float().sigmoid().cpu())
                bf.append(out['evidence'].log_bayes_factor[0].float().cpu())
            priors.append(torch.stack(p).mean(0).numpy()); fulls.append(torch.stack(f).mean(0).numpy())
            bfs.append(torch.stack(bf).mean(0).numpy())
            if (position+1)%16 == 0:
                print(json.dumps(dict(phase='fitting_readout_progress', completed=position+1, total=128)),flush=True)
    p,f,bf = map(np.stack,(priors,fulls,bfs))
    metrics = summarize(labels,masks,p,f,bf)
    private = ROOT/'readout.private.npz'
    with private.open('xb') as h:
        np.savez(h, indices=np.array(indices), context=p, full=f, logBF=bf)
    result = dict(receipt, status='fixed_final_fitting_readout_complete', metrics=metrics,
        private_arrays_sha256=sha(private), private_arrays_bytes=private.stat().st_size,
        elapsed_seconds=time.monotonic()-started, peak_GPU_bytes=torch.cuda.max_memory_allocated())
    assert sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file()) < 16*2**20
    once(ROOT/'complete.json',result)
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
