"""Compile an offline real-MRI two-T4 trainer recovery, with private inputs.

Original model/preprocessing bodies and fitting identities/targets are supplied
on authorized storage. They are never included in this repository module.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json

from scripts.kaggle.licensed_pixel_cache_probe import build as pixel_build
from scripts.kaggle.released_context_notebook import segment
from scripts.kaggle.series_boundary_notebook import PARENT_SOURCE_SHA256

MODULES = (
    "rsna/competitive/licensed_parent_train.py",
    "rsna/competitive/checkpoint.py",
    "rsna/competitive/released_context.py",
)


def build(parent, codec, files, private_targets):
    if hashlib.sha256(parent.encode()).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Exact scored licensed model source required")
    if set(files) != set(MODULES) or not isinstance(private_targets, dict):
        raise ValueError("Complete own numerical module closure and private32 targets required")
    ids = private_targets.get("uids", [])
    if len(ids) != 32 or len(set(ids)) != 32 or not private_targets.get("eligible_fitting_only"):
        raise ValueError("Exactly32 admitted fitting cases; no protected target substitution")
    for field in ("labels", "masks"):
        if len(private_targets.get(field, [])) != 32 or any(
            len(row) != 12 for row in private_targets[field]
        ):
            raise ValueError("Aligned32x12 original fitting labels and masks required")
    pixels, pixel_binding = pixel_build(parent, codec)
    suffix = "if __name__=='__main__':\n    _pixel_cache_main()\n"
    if not pixels.endswith(suffix):
        raise ValueError("Unique original renderer entry point required")
    definitions = []
    for node in ast.parse(parent).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in (
            "build_backbone",
            "RaptorClassifier",
            "load_model",
        ):
            definitions.append(segment(parent, node))
    if len(definitions) != 3:
        raise ValueError("Complete exact original encoder/MIL/strict-loader definitions required")
    model_source = "import gc,timm\nimport torch.nn as nn\n" + "\n".join(definitions)
    targets = json.dumps(private_targets, sort_keys=True, allow_nan=False, separators=(",", ":"))
    worker = (
        pixels[: -len(suffix)]
        + "\n"
        + model_source
        + r"""
import argparse,random,sys
import torch.distributed as dist
from rsna.competitive.licensed_parent_train import Recipe,fit,PARENT_SHA

def exact_index(size,sha):
    found=[]
    for directory in Path('/kaggle/input').iterdir():
        if not directory.is_dir():continue
        if any((directory/n).exists() for n in ('train.csv','test.csv','train_series','train_images')):continue
        for root,dirs,names in os.walk(directory):
            if len(Path(root).relative_to(directory).parts)>=5:dirs[:]=[]
            if 'index.private.json' in names:
                p=Path(root)/'index.private.json'
                if p.stat().st_size==size and hashlib.sha256(p.read_bytes()).hexdigest()==sha:found.append(p)
    assert found,'Original exact private32 input required'
    return found[0]

class PixelCases:
    def __init__(self,targets,indices):
        p=exact_index(21324,'3a1567989bf65779aeb19785dafdedef7386fb6bedba5ec45da6d3a67a9a79e3');index=json.loads(p.read_text());rows=index['records'];assert [r['study_uid'] for r in rows]==targets['uids'];assert index['parent_source_sha256']==_CACHE_PARENT_SHA and index['codec_sha256']==_CACHE_CODEC_SHA
        f,feature_rows=_pixel_original32();assert [r['study_uid'] for r in feature_rows]==targets['uids'];self.features=f.parent;self.frows=feature_rows;self.rows=rows;self.targets=targets;self.indices=indices;self.uids=[rows[i]['study_uid'] for i in indices];self.path=p.parent/'pixels.tar';assert self.path.stat().st_size==326799360
        with tarfile.open(self.path,'r:') as archive:self.members={m.name:m for m in archive.getmembers()}
        assert len(self.members)==32
    def __len__(self):return len(self.indices)
    def case(self,index,positions,mirror):
        i=self.indices[index];row=self.rows[i];member=self.members[row['file']];assert member.isreg() and member.size==row['bytes']
        with self.path.open('rb') as f:f.seek(member.offset_data);raw=f.read(member.size)
        volume,mask=decode(raw,row['sha256']);assert mask.any() and hashlib.sha256(volume.tobytes()).hexdigest()==row['volume_sha256'] and hashlib.sha256(mask.tobytes()).hexdigest()==row['mask_sha256'];x=eval_windows(volume,mask,94,320,norm=NORM,mirror=mirror);assert _pixel_tensor_sha(x)==row['original_mirrored_input_sha256' if mirror else 'original_plain_input_sha256']
        feature=self.frows[i];raw=(self.features/feature['file']).read_bytes();assert len(raw)==feature['bytes'] and hashlib.sha256(raw).hexdigest()==feature['sha256']
        with np.load(io.BytesIO(raw),allow_pickle=False) as a:p=a['parent_logits'].copy();assert p.shape==(2,12) and p.dtype==np.float32 and np.isfinite(p).all()
        return {'images':x[positions].contiguous(),'labels':torch.tensor(self.targets['labels'][i],dtype=torch.float32),'mask':torch.tensor(self.targets['masks'][i],dtype=torch.bool),'parent_logits':torch.from_numpy(p[int(mirror)])}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('full','stop','resume'));a=parser.parse_args();rank=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==2;torch.cuda.set_device(rank);device=torch.device('cuda',rank);assert torch.cuda.get_device_name(device)=='Tesla T4';torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);random.seed(20261016+rank);np.random.seed(20261016+rank);torch.manual_seed(20261016+rank);torch.cuda.manual_seed_all(20261016+rank);dist.init_process_group('nccl');root=Path('/kaggle/working/parent-recovery');targets=json.loads((root/'targets.private.json').read_text());weight=next(p for directory in Path('/kaggle/input').iterdir() for p in directory.rglob('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt') if p.is_file());model,res=load_model(str(weight),'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',384,device);assert res==320 and all(p.requires_grad for p in model.parameters());data=PixelCases(targets,list(range(16)));engineering=PixelCases(targets,list(range(16,32)));assert not set(data.uids)&set(engineering.uids);output=root/('full' if a.mode=='full' else 'resumed');binding={'parent_checkpoint_sha256':PARENT_SHA,'parent_source_sha256':_CACHE_PARENT_SHA,'codec_sha256':_CACHE_CODEC_SHA,'private_fitting_target_sha256':hashlib.sha256((root/'targets.private.json').read_bytes()).hexdigest(),'original32_pixel_index_sha256':'3a1567989bf65779aeb19785dafdedef7386fb6bedba5ec45da6d3a67a9a79e3','diagnostic_only':True,'independent_clinical_accuracy_claim':False};initial_bn={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if any(k in n for k in ('running_mean','running_var','num_batches_tracked'))};start=time.monotonic();result=fit(model,data,output,binding,Recipe(epochs=2),device=device,resume=a.mode=='resume',stop_after=1 if a.mode=='stop' else None)
    assert all(torch.equal(v,model.state_dict()[n].detach().cpu()) for n,v in initial_bn.items());pred=[]
    if a.mode!='stop':
        model.eval()
        with torch.no_grad():
            for i in range(rank,16,2):
                views=[]
                for mirror in (False,True):
                    x=engineering.case(i,list(range(94)),mirror)['images'].to(device)
                    with torch.autocast('cuda',dtype=torch.float16):z=model(x.unsqueeze(0)).float()
                    assert z.shape==(1,12) and torch.isfinite(z).all();views.append(z.sigmoid().cpu())
                pred.append((i,torch.stack(views).mean(0)))
        gathered=[None,None];dist.all_gather_object(gathered,pred)
        if rank==0:
            pairs=sorted(x for group in gathered for x in group);assert [i for i,p in pairs]==list(range(16));values=torch.cat([p for i,p in pairs]).numpy();np.save(output/'engineering-predictions.private.npy',values,allow_pickle=False)
    info={'rank':rank,'GPU':torch.cuda.get_device_name(device),'total_VRAM_bytes':torch.cuda.get_device_properties(device).total_memory,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'finite_updates':result['completed'],'elapsed_seconds':time.monotonic()-start,'status':result['status'],'all_BN_buffers_equal_parent':True,'trainable_parameters':sum(p.numel() for p in model.parameters()),'CPU_threads_per_rank':torch.get_num_threads(),'no_independent_accuracy_claim':True};output.mkdir(exist_ok=True);(output/(a.mode+'-rank-'+str(rank)+'.json')).write_text(json.dumps(info,sort_keys=True));dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
"""
    )
    compile(worker, "private_original_MRI_twoT4_worker", "exec")
    bootstrap = r"""
import os,sys,json,base64,subprocess,time,hashlib,shutil
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
FILES=FILES_VALUE
WORKER=WORKER_VALUE
TARGETS=TARGETS_VALUE
root=Path('/kaggle/working/parent-recovery');assert not root.exists();assert shutil.disk_usage('/kaggle/working').free>8*2**30;root.mkdir(mode=0o700);code=root/'code';code.mkdir(mode=0o700)
for name,text in {**FILES,'rsna/__init__.py':'','rsna/competitive/__init__.py':''}.items():
    p=code/name;assert p.resolve().is_relative_to(code.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);p.chmod(0o600)
(root/'targets.private.json').write_bytes(base64.b64decode(TARGETS));(root/'targets.private.json').chmod(0o600);worker=code/'worker.private.py';worker.write_bytes(base64.b64decode(WORKER));worker.chmod(0o600);env={**os.environ,'PYTHONPATH':str(code)};start=time.monotonic();phases=[]
for phase in ('full','stop','resume'):
    with (root/(phase+'.private.log')).open('x') as log:
        result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',str(worker),phase],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
    assert result.returncode==0,'Original real-MRI recovery process failed; preserve log/checkpoints without retry';phases.append({'phase':phase,'elapsed_from_start_seconds':time.monotonic()-start})
import torch
full=torch.load(root/'full/checkpoint.pt',map_location='cpu',weights_only=True);resumed=torch.load(root/'resumed/checkpoint.pt',map_location='cpu',weights_only=True)
def same(a,b):
    if isinstance(a,torch.Tensor):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b
same(full,resumed);assert full['completed']==4 and full['binding']['world_size']==2;pred=(root/'full/engineering-predictions.private.npy').read_bytes();assert pred==(root/'resumed/engineering-predictions.private.npy').read_bytes();devices=[json.loads((root/'full'/('full-rank-'+str(rank)+'.json')).read_text()) for rank in (0,1)];assert all(x['finite_updates']==4 and x['all_BN_buffers_equal_parent'] for x in devices);assert full['history'][0]['gradient_norm']>0;files_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());assert files_bytes<6*2**30
receipt={'status':'actual_original_MRI_parent_twoT4_full_stop1_resume_complete_state_and_predictions_accepted','source_mode':'diagnostic_only_not_competition_model','fit_studies':16,'disjoint_engineering_studies':16,'epochs':2,'finite_updates':4,'full_learned_encoder_MIL_AdamW_LR_cursors_all_rank_RNG_and_GradScaler_exact':True,'all16_engineering_two_view_prediction_bytes_equal':True,'all_BN_buffers_equal_parent':True,'world_size':2,'devices':devices,'source_module_sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in FILES.items()},'private_target_sha256':hashlib.sha256(base64.b64decode(TARGETS)).hexdigest(),'worker_source_sha256':hashlib.sha256(base64.b64decode(WORKER)).hexdigest(),'parent_weights_sha256':'7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef','private_prediction_sha256':hashlib.sha256(pred).hexdigest(),'all_retained_bytes':files_bytes,'elapsed_seconds':time.monotonic()-start,'phases':phases,'clinical_or_competition_accuracy_claim':False,'full_fit_dispatched':False,'competition_entry':False};(Path('/kaggle/working')/'licensed-parent-real-MRI-recovery-accepted.json').write_text(json.dumps(receipt,sort_keys=True,indent=2));print(json.dumps(receipt),flush=True)
"""
    for name, value in {
        "FILES_VALUE": repr(files),
        "WORKER_VALUE": repr(base64.b64encode(worker.encode()).decode()),
        "TARGETS_VALUE": repr(base64.b64encode(targets.encode()).decode()),
    }.items():
        bootstrap = bootstrap.replace(name, value)
    compile(bootstrap, "private_original_MRI_twoT4_recovery", "exec")
    return bootstrap, {
        "source_sha256": hashlib.sha256(bootstrap.encode()).hexdigest(),
        "source_bytes": len(bootstrap.encode()),
        "worker_source_sha256": hashlib.sha256(worker.encode()).hexdigest(),
        "own_module_sha256": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in files.items()},
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "pixel_binding": pixel_binding,
        "private_target_sha256": hashlib.sha256(targets.encode()).hexdigest(),
        "fitting_studies": 16,
        "disjoint_engineering_studies": 16,
        "GPU_world_size": 2,
        "competition_entry": False,
    }
