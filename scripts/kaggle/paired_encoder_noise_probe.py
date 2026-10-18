"""Compile one new trained-encoder sampling-noise measurement; never a fit or entry."""

from __future__ import annotations

import ast
import base64
import hashlib

from rsna.competitive.licensed_importance_serving import provenance
from scripts.kaggle.paired_logit_recovery import literals

RECOVERY_SOURCE = "4b129e1c5c76bcc226eff30f30e9e59b25df0ba91358771b795f7d65f68ad6b5"
ADAPTED_MODEL = "f6a2007e7264fe5021cba4a4b58c0897a1e20f181078b7f566962ea10b2a04f4"
EXTRA = (
    "rsna/competitive/paired_encoder_noise.py",
    "rsna/competitive/paired_encoder_probe.py",
    "rsna/competitive/licensed_importance_serving.py",
)
STATUS = "actual_trained_encoder_FP16_dropout_paired_sampling_noise_measured"


def build(recovery_source, files, training, publication):
    if (
        hashlib.sha256(recovery_source.encode()).hexdigest() != RECOVERY_SOURCE
        or set(files) != set(EXTRA)
        or training.get("kernel_id") != 139220032
        or training.get("role") != "influence"
        or training.get("epoch") != 3
        or training.get("source_sha256")
        != "cff5c493ee2832a0672298f1942662474aa3a01d0b421069c4db171ed394b012"
        or provenance(training, publication, ADAPTED_MODEL) != 3
    ):
        raise ValueError("Exact new recovery source and preserved original influence-E3 required")
    values = literals(recovery_source)
    old_worker = base64.b64decode(values["WORKER"], validate=True).decode()
    main = [
        n for n in ast.parse(old_worker).body if isinstance(n, ast.FunctionDef) and n.name == "main"
    ]
    if len(main) != 1:
        raise ValueError("One original worker entry point required")
    prefix = "".join(old_worker.splitlines(keepends=True)[: main[0].lineno - 1])
    binding = {
        "training": training,
        "publication": publication,
        "model_sha256": ADAPTED_MODEL,
        "model_bytes": publication["files"]["model.pt"]["bytes"],
        "model_filename": "epoch-003.pt",
        "fit_cases": 4,
        "reference_draws_per_view": 16,
        "paired_draws_per_view": 32,
        "same_original_parent_and_fitting32_pixel_closure": True,
        "source_recovery_sha256": RECOVERY_SOURCE,
        "competition_entries": 0,
        "actual_optimizer_timing_or_clinical_acceptance": False,
    }
    worker = (
        prefix
        + "\nBINDING="
        + repr(binding)
        + r"""

def _locate_adapted():
    candidates=[]
    for directory in Path('/kaggle/input').iterdir():
        if not directory.is_dir():continue
        for p in directory.rglob('epoch-003.pt'):
            if p.is_file() and not p.is_symlink() and p.stat().st_size==BINDING['model_bytes']:
                with p.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                if digest==BINDING['model_sha256']:candidates.append(p)
    assert len(candidates)==1,'One exact previously preserved trained E3 input required'
    return candidates[0]


def main():
    import copy,gc
    from rsna.competitive.licensed_importance_serving import validate
    from rsna.competitive.mil_sampling import readout_moments,proposal
    from rsna.competitive.paired_encoder_probe import measure_case
    rank=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==2 and rank in (0,1)
    torch.cuda.set_device(rank);device=torch.device('cuda',rank);assert torch.cuda.get_device_name(device)=='Tesla T4'
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    root=Path('/kaggle/working/paired-encoder-noise-v1');targets=json.loads((root/'targets.private.json').read_text());original=PixelCases(targets,list(range(4)))
    weight=next(p for directory in Path('/kaggle/input').iterdir() for p in directory.rglob('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt') if p.is_file());base,res=load_model(str(weight),'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',384,device);assert res==320
    norm=copy.deepcopy(base.norm).cpu().float().eval();att=copy.deepcopy(base.att).cpu().float().eval();clsw=base.clsW.detach().cpu().float().clone();clsb=base.clsb.detach().cpu().float().clone()
    for module in (norm,att):
        for p in module.parameters():p.requires_grad_(False)
    names=('norm.weight','norm.bias','att.0.weight','att.0.bias','att.3.weight','att.3.bias','clsW','clsb');parent_head={'format':'licensed_parent_MIL_readout_v1','parent_checkpoint_sha256':PARENT_SHA,'source_notebook_sha256':_CACHE_PARENT_SHA,'model':{n:base.state_dict()[n].detach().cpu().float().clone() for n in names}}
    def parent_readout(features,logq):
        h=norm(features);scores=att(h).float()
        if logq is not None:scores=scores-logq[None,:,None]
        pooled=torch.einsum('bkn,bkf->bnf',scores.softmax(1),h)
        return (pooled*clsw).sum(-1)+clsb
    model=_whole_model(base);del base;path=_locate_adapted();state=torch.load(path,map_location='cpu',weights_only=True);actual=validate(state,BINDING['training'],BINDING['publication'],BINDING['model_sha256']);assert actual['epoch']==3;model.load_state_dict(state['model'],strict=True);assert sum(p.numel() for p in model.parameters())==73136096 and all(p.requires_grad for p in model.parameters());gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    for index in range(rank,4,2):
        row=original.frows[index];raw=(original.features/row['file']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:features=torch.from_numpy(arrays['features'].copy())
        weights,votes,bias,_=readout_moments(parent_head,features);q=torch.from_numpy(proposal(weights,votes,bias));views=[original.case(index,list(range(94)),mirror) for mirror in (False,True)]
        def progress(event):
            event={'case_index':index,'rank':rank,**event};(root/('progress-rank-'+str(rank)+'.json')).write_text(json.dumps(event));print(json.dumps(event),flush=True)
        result=measure_case(model,views,features,parent_readout,q,original.uids[index],Recipe(role='influence'),device=device,expected_parameters=73136096,reference_draws=16,paired_draws=32,progress=progress)
        result.update(case_index=index,rank=rank);(root/('case-'+str(index)+'.private.json')).write_text(json.dumps(result,indent=2));del views,features;gc.collect();torch.cuda.empty_cache()
    assert all(torch.equal(t,model.state_dict()[name].detach().cpu()) for name,t in state['model'].items()),'New probe must never update source checkpoint tensors'
    device_result={'rank':rank,'GPU':torch.cuda.get_device_name(device),'parameters':73136096,'completed_cases':2,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'total_VRAM_bytes':torch.cuda.get_device_properties(device).total_memory,'original_trained_complete_state_unchanged':True,'CPU_threads':torch.get_num_threads()};assert 0<device_result['peak_allocated_bytes']<device_result['total_VRAM_bytes'];(root/('device-'+str(rank)+'.json')).write_text(json.dumps(device_result))

if __name__=='__main__':main()
"""
    )
    worker = worker.replace("paired-logit-recovery", "paired-encoder-noise-v1")
    compile(worker, "new_trained_encoder_noise_worker", "exec")
    closure = {**values["FILES"], **files}
    bootstrap = r"""
import os,sys,json,base64,subprocess,hashlib,shutil,time
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
FILES=FILES_VALUE
WORKER=WORKER_VALUE
TARGETS=TARGETS_VALUE
PERMISSION=PERMISSION_VALUE
BINDING=BINDING_VALUE
root=Path('/kaggle/working/paired-encoder-noise-v1');assert not root.exists() and shutil.disk_usage('/kaggle/working').free>4*2**30;root.mkdir(mode=0o700);code=root/'code';code.mkdir(mode=0o700)
for name,text in {**FILES,'rsna/__init__.py':'','rsna/competitive/__init__.py':''}.items():
    p=code/name;assert p.resolve().is_relative_to(code.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);p.chmod(0o600)
(root/'targets.private.json').write_bytes(base64.b64decode(TARGETS));(root/'source-permission.txt').write_bytes(base64.b64decode(PERMISSION));worker=code/'worker.private.py';worker.write_bytes(base64.b64decode(WORKER));env={**os.environ,'PYTHONPATH':str(code)};start=time.monotonic()
with (root/'new-noise-probe.private.log').open('x') as log:
    out=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',str(worker)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1650)
assert out.returncode==0,'New actual trained-encoder noise trial stopped; retain originals, no automatic retry'
rows=[json.loads((root/('case-'+str(i)+'.private.json')).read_text()) for i in range(4)];views=[v for row in rows for v in row['views']];assert len(views)==8 and all(r['parameters']==73136096 and r['optimizer_or_parameter_updates']==0 for r in rows)
groups={}
for group in ('all','encoder','head'):
    values=[v['groups'][group] for v in views];n=values[0]['parameters'];assert all(v['parameters']==n for v in values)
    stats={arm:{key:sum(v[arm][key] for v in values)/8 for key in ('MSE_against_estimated_dense_gradient','empirical_mean_gradient_error','MSE_minus_estimated_reference_noise')} for arm in ('ordinary','controlled')};se=sum(v['reference_mean_squared_standard_error'] for v in values)/8
    ratio=lambda a,b:a/b if b>0 else None
    groups[group]={'parameters':n,**stats,'reference_mean_squared_standard_error':se,'MSE_ratio':ratio(stats['controlled']['MSE_against_estimated_dense_gradient'],stats['ordinary']['MSE_against_estimated_dense_gradient']),'within_case_view_empirical_mean_error_ratio':ratio(stats['controlled']['empirical_mean_gradient_error'],stats['ordinary']['empirical_mean_gradient_error']),'reference_noise_fraction_of_ordinary_MSE':ratio(se,stats['ordinary']['MSE_against_estimated_dense_gradient'])}
forward={arm:sum(v['forward_MSE'][arm] for v in views)/8 for arm in ('ordinary','controlled')};timing={arm:sum(v['sampled_forward_backward_seconds'][arm] for v in views) for arm in ('ordinary','controlled')};devices=[json.loads((root/('device-'+str(i)+'.json')).read_text()) for i in (0,1)];assert groups['all']['parameters']==groups['encoder']['parameters']+groups['head']['parameters']==73136096;assert all(d['original_trained_complete_state_unchanged'] for d in devices);retained=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());assert retained<16*2**20 and not Path('/kaggle/working/submission.csv').exists()
receipt={'status':'STATUS_VALUE','fit_cases':4,'views_per_case':2,'loss_averages_views':False,'dense_reference_draws_per_view':16,'paired_sampled_draws_per_view':32,'gradients':'all_learned_parameters_preclip_per_study_unscaled_FP32','forward_precision':'actual_FP16_autocast','gradient_scale':1024.0,'groups':groups,'forward_probability_MSE':forward,'forward_probability_MSE_ratio':forward['controlled']/forward['ordinary'] if forward['ordinary']>0 else None,'forward_backward_timing_seconds':timing,'forward_backward_timing_ratio':timing['controlled']/timing['ordinary'],'optimizer_step_time_or_proven_gradient_bias_accepted':False,'clinical_accuracy_or_production_fit_accepted':False,'actual_encoder_recovery_is_separate':True,'devices':devices,'trained_model_sha256':BINDING['model_sha256'],'trained_source_sha256':BINDING['training']['source_sha256'],'own_module_sha256':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in FILES.items()},'worker_sha256':hashlib.sha256(base64.b64decode(WORKER)).hexdigest(),'private_fitting_targets_sha256':hashlib.sha256(base64.b64decode(TARGETS)).hexdigest(),'elapsed_seconds':time.monotonic()-start,'retained_bytes':retained,'competition_entries':0};(Path('/kaggle/working')/'paired-encoder-noise-measured.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)
"""
    for name, value in {
        "FILES_VALUE": closure,
        "WORKER_VALUE": base64.b64encode(worker.encode()).decode(),
        "TARGETS_VALUE": values["TARGETS"],
        "PERMISSION_VALUE": values["PERMISSION"],
        "BINDING_VALUE": binding,
    }.items():
        bootstrap = bootstrap.replace(name, repr(value))
    bootstrap = bootstrap.replace("STATUS_VALUE", STATUS)
    compile(bootstrap, "new_trained_encoder_noise_bootstrap", "exec")
    return bootstrap, {
        "source_sha256": hashlib.sha256(bootstrap.encode()).hexdigest(),
        "worker_sha256": hashlib.sha256(worker.encode()).hexdigest(),
        "own_module_sha256": {
            n: hashlib.sha256(s.encode()).hexdigest() for n, s in closure.items()
        },
        "binding": binding,
        "required_actual_recovery_kernel": 139417155,
        "GPU_session_cap_seconds": 1800,
        "output_cap_bytes": 16 * 2**20,
        "model_or_pixel_Dataset_created": False,
        "competition_entries": 0,
    }
