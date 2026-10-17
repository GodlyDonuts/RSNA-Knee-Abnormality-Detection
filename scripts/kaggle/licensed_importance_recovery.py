"""Compile one matched, real-MRI sampled-encoder recovery on two T4s.

The existing licensed renderer/model body and32 private fitting engineering
targets are supplied on authorized storage. No external body or records here.
This is not a competition candidate and cannot dispatch a production fit.
"""

from __future__ import annotations

import ast
import base64
import hashlib

from scripts.kaggle.licensed_parent_recovery import MODULES, WHOLE_MODULE
from scripts.kaggle.licensed_parent_recovery import build as original_build

EXTRA_MODULES = (
    "rsna/competitive/licensed_importance.py",
    "rsna/competitive/licensed_importance_train.py",
    "rsna/competitive/mil_sampling.py",
    "rsna/competitive/released_context_train.py",
    "rsna/competitive/released_context_infer.py",
    "rsna/schema.py",
)
PERMISSION_SHA = "34c44b20e4d7ace4eb523b0423b36d36dec826fe6ae54c66346796bba0190e79"


def build(parent, codec, files, private_targets, *, permission):
    expected = {*MODULES, WHOLE_MODULE, *EXTRA_MODULES}
    if set(files) != expected:
        raise ValueError("Complete matched sampled-encoder numerical closure required")
    if (
        not isinstance(permission, str)
        or hashlib.sha256(permission.encode()).hexdigest() != PERMISSION_SHA
    ):
        raise ValueError("Exact existing source/adaptation permission must accompany the model")
    original, binding = original_build(
        parent,
        codec,
        {n: files[n] for n in (*MODULES, WHOLE_MODULE)},
        private_targets,
        whole_study=True,
    )
    literals = {}
    for node in ast.parse(original).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in ("WORKER", "TARGETS"):
                literals[target.id] = ast.literal_eval(node.value)
    if set(literals) != {"WORKER", "TARGETS"}:
        raise ValueError("Exact private original renderer/target worker required")
    worker = base64.b64decode(literals["WORKER"], validate=True).decode()

    def change(old, new):
        nonlocal worker
        if worker.count(old) != 1:
            raise ValueError("Original source template changed; review before GPU")
        worker = worker.replace(old, new)

    change(
        "class WholeStudyRaptor(CheckpointedEncode,RaptorClassifier):pass",
        "from rsna.competitive.licensed_importance import ImportanceReadout\n"
        "class WholeStudyRaptor(ImportanceReadout,CheckpointedEncode,RaptorClassifier):pass",
    )
    change(
        "from rsna.competitive.licensed_parent_train import Recipe,fit,PARENT_SHA",
        "from rsna.competitive.licensed_parent_train import PARENT_SHA\n"
        "from rsna.competitive.licensed_importance import Recipe,ProposalCases\n"
        "from rsna.competitive.licensed_importance_train import fit",
    )
    helper = r"""
def _teacher_proposals(model,data):
    from rsna.competitive.mil_sampling import readout_moments,proposal
    names=('norm.weight','norm.bias','att.0.weight','att.0.bias','att.3.weight','att.3.bias','clsW','clsb')
    state=model.state_dict();head={'format':'licensed_parent_MIL_readout_v1','parent_checkpoint_sha256':PARENT_SHA,'source_notebook_sha256':_CACHE_PARENT_SHA,'model':{n:state[n].detach().cpu().float().clone() for n in names}}
    out=[]
    for i in data.indices:
        row=data.frows[i];raw=(data.features/row['file']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:features=torch.from_numpy(arrays['features'].copy())
        assert features.shape==(2,94,1024) and features.dtype==torch.float32
        weights,votes,bias,_=readout_moments(head,features);out.append(torch.from_numpy(proposal(weights,votes,bias)))
    return torch.stack(out)

def _initial_identity(model,device):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        value=value.detach().cpu().contiguous();h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    state=np.random.get_state()
    return {'model':h.hexdigest(),'python_rng':hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),'numpy_rng':hashlib.sha256(state[1].tobytes()+repr((state[0],*state[2:])).encode()).hexdigest(),'torch_cpu_rng':hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),'torch_cuda_rng':hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()}

"""
    change("def main():\n", helper + "def main():\n")
    change(
        "a=parser.parse_args();",
        "parser.add_argument('role',choices=('uniform','influence'));a=parser.parse_args();",
    )
    change(
        "data=PixelCases(targets,list(range(16)));engineering=",
        "original_data=PixelCases(targets,list(range(16)));data=ProposalCases(original_data,_teacher_proposals(model,original_data));initial_identity=_initial_identity(model,device);engineering=",
    )
    change(
        "output=root/('full' if a.mode=='full' else 'resumed')",
        "output=root/a.role/('full' if a.mode=='full' else 'resumed')",
    )
    change("Recipe(epochs=2,windows=94)", "Recipe(epochs=2,role=a.role)")
    change(
        "'training_bag':'all94_original_windows_checkpointed8'",
        "'training_bag':'matched_IID16_inverse_proposal_frozen_teacher_v1'",
    )
    change(
        "'rank':rank,'GPU':torch.cuda.get_device_name(device),'total_VRAM_bytes'",
        "'initial_identity':initial_identity,'rank':rank,'GPU':torch.cuda.get_device_name(device),'total_VRAM_bytes'",
    )
    compile(worker, "private_sampled_encoder_real_MRI_worker", "exec")
    bootstrap = r"""
import os,sys,json,base64,subprocess,time,hashlib,shutil
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
FILES=FILES_VALUE
WORKER=WORKER_VALUE
TARGETS=TARGETS_VALUE
PERMISSION=PERMISSION_VALUE
root=Path('/kaggle/working/importance-recovery');assert not root.exists() and shutil.disk_usage('/kaggle/working').free>12*2**30;root.mkdir(mode=0o700);code=root/'code';code.mkdir(mode=0o700)
(root/'source-permission.txt').write_bytes(base64.b64decode(PERMISSION))
weights=[p for d in Path('/kaggle/input').iterdir() if d.is_dir() for p in d.rglob('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt') if p.is_file()];assert len(weights)==1
with weights[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()=='7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
for name,text in {**FILES,'rsna/__init__.py':'','rsna/competitive/__init__.py':''}.items():
    p=code/name;assert p.resolve().is_relative_to(code.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);p.chmod(0o600)
(root/'targets.private.json').write_bytes(base64.b64decode(TARGETS));(root/'targets.private.json').chmod(0o600);worker=code/'worker.private.py';worker.write_bytes(base64.b64decode(WORKER));worker.chmod(0o600)
env={**os.environ,'PYTHONPATH':str(code)};start=time.monotonic();phases=[]
for role in ('uniform','influence'):
    for phase in ('full','stop','resume'):
        with (root/(role+'-'+phase+'.private.log')).open('x') as log:
            result=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',str(worker),phase,role],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        assert result.returncode==0,'Original sampled-encoder process failed; retain all logs/checkpoints, no retry';phases.append({'role':role,'phase':phase,'elapsed_seconds':time.monotonic()-start});print(json.dumps(phases[-1]),flush=True)
import torch
def same(a,b):
    if isinstance(a,torch.Tensor):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for name in a:same(a[name],b[name])
    elif isinstance(a,(list,tuple)):
        assert type(a) is type(b) and len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b
roles={};identities=[];finals=[]
for role in ('uniform','influence'):
    d=root/role;full=torch.load(d/'full/checkpoint.pt',map_location='cpu',weights_only=True);resumed=torch.load(d/'resumed/checkpoint.pt',map_location='cpu',weights_only=True);same(full,resumed);assert full['completed']==4 and full['binding']['world_size']==2 and full['binding']['recipe']['windows']==16 and full['binding']['recipe']['role']==role;assert full['binding']['teacher_proposal_sha256']==resumed['binding']['teacher_proposal_sha256'];assert [x['update'] for x in full['history']]==[1,2,3,4]
    for epoch in (1,2):same(torch.load(d/'full/epoch-checkpoints'/('epoch-'+str(epoch).zfill(3)+'.pt'),map_location='cpu',weights_only=True),torch.load(d/'resumed/epoch-checkpoints'/('epoch-'+str(epoch).zfill(3)+'.pt'),map_location='cpu',weights_only=True))
    pred=(d/'full/engineering-predictions.private.npy').read_bytes();assert pred==(d/'resumed/engineering-predictions.private.npy').read_bytes();devices=[json.loads((d/'full'/('full-rank-'+str(rank)+'.json')).read_text()) for rank in (0,1)];assert all(x['GPU']=='Tesla T4' and x['finite_updates']==4 and x['trainable_parameters']==73136096 and x['all_BN_buffers_equal_parent'] for x in devices);assert all(x['gradient_norm']>0 and x['loss']>=0 for x in full['history']);identities.append([x['initial_identity'] for x in devices]);finals.append(full['model']);roles[role]={'complete_model_AdamW_LR_cursors_allrank_RNG_scaler_history_and_epoch_snapshots_equal':True,'all16_full94_two_view_prediction_bytes_equal':True,'prediction_SHA':hashlib.sha256(pred).hexdigest(),'proposal_SHA':full['binding']['teacher_proposal_sha256'],'finite_updates':4,'devices':devices}
same(identities[0],identities[1]);assert roles['uniform']['proposal_SHA']==roles['influence']['proposal_SHA'];distinct=sum(not torch.equal(finals[0][n],v) for n,v in finals[1].items());assert distinct>0,'Actual proposal must cause distinct learned tensors to justify full comparison';files_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());assert files_bytes<12*2**30 and not Path('/kaggle/working/submission.csv').exists()
receipt={'status':'actual_matched_uniform_influence_full_encoder_twoT4_MRI_recovery_accepted','source_mode':'diagnostic_only_not_competition_model','fit_studies':16,'disjoint_engineering_studies':16,'epochs':2,'updates_per_role':4,'sampled_windows':16,'full_inference_windows':94,'proposal_uniform_fraction':0.25,'world_size':2,'roles':roles,'same_initial_complete_model_and_allrank_RNG_between_roles':True,'distinct_final_learned_tensors':distinct,'source_module_sha256':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in FILES.items()},'worker_source_sha256':hashlib.sha256(base64.b64decode(WORKER)).hexdigest(),'private_target_sha256':hashlib.sha256(base64.b64decode(TARGETS)).hexdigest(),'parent_weights_sha256':'7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef','all_retained_bytes':files_bytes,'elapsed_seconds':time.monotonic()-start,'phases':phases,'clinical_or_competition_accuracy_claim':False,'full_production_fit_dispatched':False,'competition_entry':False};(Path('/kaggle/working')/'licensed-importance-real-MRI-recovery-accepted.json').write_text(json.dumps(receipt,sort_keys=True,indent=2));print(json.dumps(receipt),flush=True)
"""
    # The original pixel-reader uses this root for exact feature lookup as well.
    worker = worker.replace(
        "Path('/kaggle/working/parent-recovery')", "Path('/kaggle/working/importance-recovery')"
    )
    for name, value in {
        "FILES_VALUE": files,
        "WORKER_VALUE": base64.b64encode(worker.encode()).decode(),
        "TARGETS_VALUE": literals["TARGETS"],
        "PERMISSION_VALUE": base64.b64encode(permission.encode()).decode(),
    }.items():
        bootstrap = bootstrap.replace(name, repr(value))
    compile(bootstrap, "private_matched_sampled_encoder_recovery", "exec")
    return bootstrap, {
        "source_sha256": hashlib.sha256(bootstrap.encode()).hexdigest(),
        "worker_sha256": hashlib.sha256(worker.encode()).hexdigest(),
        "own_module_sha256": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in files.items()},
        "private_target_sha256": binding["private_target_sha256"],
        "parent_source_sha256": binding["parent_source_sha256"],
        "permission_sha256": PERMISSION_SHA,
        "fit_studies": 16,
        "disjoint_engineering_studies": 16,
        "roles": ["uniform", "influence"],
        "maximum_retained_GPU_bytes": 12 * 2**30,
        "competition_entries": 0,
        "full_production_fit_dispatched": False,
    }
