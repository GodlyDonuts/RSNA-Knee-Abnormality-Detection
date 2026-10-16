"""Compile one checkpoint-continuous whole94 full-parent training session.

The accepted private recovery supplies the attributed parent body. No external
source body, weights, patient arrays or competition submissions are embedded here.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json

from rsna.competitive.licensed_parent_population import validate_proofs

RECOVERY_SHA = "a45d2a00322d84e412c4bd28235fc49d81a42f3ec3e733f9b7da76f68ec57b39"
WORKER_SHA = "9f87ffd3b4a3a090817a42fe0ee43161367efb582784db0a2176df2453606791"
ACCEPTED_MODULES = {
    "rsna/competitive/licensed_parent_train.py": "cf238d9154806a51cd92c80a61e39ca344ef865558305db51cbfdc86f89eeec0",
    "rsna/competitive/licensed_full_bag.py": "74a4e19eb37570f66754fa3b4d8e2e117fadb7cc91c61345efc4e962cbf3983f",
    "rsna/competitive/checkpoint.py": "a70ef871e200bac89e4f071d9394ce8b0abc1c2ba0017eaba3692d25dde31e21",
    "rsna/competitive/released_context.py": "4ed7cd1978a4acac6c233c734914dbe713aed818f229eabd7f23fa10580fbee9",
    "rsna/competitive/licensed_pixel_cache.py": "7bdaf41a864b499401ff4a62e6288d30eead3849c0fda5c49cc67b3b2d0e2ab5",
    "rsna/competitive/licensed_pixel_bags.py": "857a374f9f6f88b10028b6a899961d6c62c64151f800710deaa20a915cb5616e",
}
POPULATION_MODULE = "rsna/competitive/licensed_parent_population.py"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def worker_source(accepted, files, proofs, source_git):
    validate_proofs(proofs)
    if digest(accepted) != RECOVERY_SHA or set(files) != {*ACCEPTED_MODULES, POPULATION_MODULE}:
        raise ValueError("Exact accepted whole94 recovery and complete numerical closure required")
    if any(digest(files[n]) != h for n, h in ACCEPTED_MODULES.items()):
        raise ValueError("Actual accepted numerical modules and CPU input bridge must remain exact")
    if len(source_git) != 40 or any(c not in "0123456789abcdef" for c in source_git):
        raise ValueError("Immutable production source Git required")
    nodes = [
        n
        for n in ast.parse(accepted).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "WORKER" for t in n.targets)
    ]
    if len(nodes) != 1:
        raise ValueError("One original accepted private worker required")
    source = base64.b64decode(ast.literal_eval(nodes[0].value), validate=True).decode()
    if digest(source) != WORKER_SHA:
        raise ValueError("Exact original actual-T4 worker required")
    main_nodes = [
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "main"
    ]
    if len(main_nodes) != 1:
        raise ValueError("One original worker entry point required")
    binding = {
        "production_plan": "licensed_MRI_parent_all94_full3887_fixed3_continuous_v1",
        "source_git": source_git,
        "parent_checkpoint_sha256": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef",
        "parent_source_sha256": "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde",
        "source_module_sha256": {n: digest(s) for n, s in sorted(files.items())},
        "actual_whole94_recovery": {
            "kernel_id": 139105348,
            "version": 1,
            "source_sha256": RECOVERY_SHA,
            "worker_sha256": WORKER_SHA,
        },
        "actual_PixelBags_CPU_bridge": "939362",
        "original_target_dataset": {
            "id": 12641441,
            "version": 1,
            "sha256": "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285",
        },
        "pixel_collection_proofs_sha256": digest(
            json.dumps(proofs, sort_keys=True, separators=(",", ":"))
        ),
        "training_bag": "all94_original_windows_checkpointed8",
        "diagnostic_only": False,
        "independent_internal_validation": False,
        "protected_gold_confirmation_used": False,
        "clinical_or_competition_accuracy_claim": False,
    }
    main = r"""
from rsna.competitive.licensed_parent_population import load as load_population
_PRODUCTION_PROOFS=PROOFS_VALUE
_PRODUCTION_BINDING=BINDING_VALUE

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--stop-after',type=int);a=parser.parse_args();assert a.stop_after in (None,486,972);rank=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==2;torch.cuda.set_device(rank);device=torch.device('cuda',rank);assert torch.cuda.get_device_name(device)=='Tesla T4';torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);random.seed(20261016+rank);np.random.seed(20261016+rank);torch.manual_seed(20261016+rank);torch.cuda.manual_seed_all(20261016+rank);dist.init_process_group('nccl');root=Path('/kaggle/working/licensed-parent-full');render=lambda volume,mask,mirror:eval_windows(volume,mask,94,320,norm=NORM,mirror=mirror);data=load_population('/kaggle/input',_PRODUCTION_PROOFS,render);assert len(data)==3887
    weight=next(p for directory in Path('/kaggle/input').iterdir() for p in directory.rglob('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt') if p.is_file());base,res=load_model(str(weight),'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',384,device);model=_whole_model(base);del base;assert res==320 and all(p.requires_grad for p in model.parameters()) and sum(p.numel() for p in model.parameters())==73136096;output=root/'fit';initial_bn={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if any(k in n for k in ('running_mean','running_var','num_batches_tracked'))};output.mkdir(parents=True,exist_ok=True);(root/('allocated-rank-'+str(rank)+'.json')).write_text(json.dumps({'rank':rank,'GPU':torch.cuda.get_device_name(device),'studies':len(data),'trainable_parameters':73136096,'original_inputs_accepted':True,'resume_requested':a.resume}));start=time.monotonic();result=fit(model,data,output,_PRODUCTION_BINDING,Recipe(epochs=3,windows=94),device=device,resume=a.resume,stop_after=a.stop_after)
    assert result['total']==1458 and result['completed'] in (486,972,1458) and all(torch.equal(v,model.state_dict()[n].detach().cpu()) for n,v in initial_bn.items());info={'rank':rank,'GPU':torch.cuda.get_device_name(device),'total_VRAM_bytes':torch.cuda.get_device_properties(device).total_memory,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'completed':result['completed'],'total':result['total'],'elapsed_seconds':time.monotonic()-start,'status':result['status'],'all_BN_buffers_equal_parent':True,'trainable_parameters':sum(p.numel() for p in model.parameters()),'CPU_threads_per_rank':torch.get_num_threads(),'no_independent_accuracy_claim':True};(output/('actual-rank-'+str(rank)+'.json')).write_text(json.dumps(info,sort_keys=True));dist.barrier();dist.destroy_process_group()
""".replace("PROOFS_VALUE", repr(proofs)).replace("BINDING_VALUE", repr(binding))
    node = main_nodes[0]
    lines = source.splitlines(keepends=True)
    source = "".join(lines[: node.lineno - 1]) + main + "\n" + "".join(lines[node.end_lineno :])
    compile(source, "private_accepted_whole94_full_production_worker", "exec")
    return source, binding


def build(accepted, files, proofs, source_git, segment, *, previous=None):
    if type(segment) is not int or not 1 <= segment <= 3:
        raise ValueError("One of three fixed epoch-boundary sessions required")
    worker, binding = worker_source(accepted, files, proofs, source_git)
    if segment == 1 and previous is not None:
        raise ValueError("First session starts only from the accepted MRI parent")
    if segment > 1:
        if (
            not isinstance(previous, dict)
            or previous.get("status") != "actual_whole94_full_parent_epoch_session_complete"
            or previous.get("segment") != segment - 1
            or previous.get("completed") != 486 * (segment - 1)
            or previous.get("worker_sha256") != digest(worker)
            or previous.get("binding") != binding
        ):
            raise ValueError(
                "Exact previous completed session and same complete training binding required"
            )
        for name in ("checkpoint", "run"):
            item = previous.get(name, {})
            if (
                type(item.get("bytes")) is not int
                or not 0 < item["bytes"] <= 2 * 2**30
                or not isinstance(item.get("sha256"), str)
                or len(item["sha256"]) != 64
            ):
                raise ValueError("Exact previous checkpoint and run whole-byte receipts required")
    bootstrap = r"""
import os,sys,json,base64,subprocess,time,hashlib,shutil
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
FILES=FILES_VALUE
WORKER=WORKER_VALUE
BINDING=BINDING_VALUE
SEGMENT=SEGMENT_VALUE
PREVIOUS=PREVIOUS_VALUE
root=Path('/kaggle/working/licensed-parent-full');assert not root.exists() and shutil.disk_usage('/kaggle/working').free>8*2**30;root.mkdir(mode=0o700);code=root/'code';code.mkdir(mode=0o700)
for name,text in {**FILES,'rsna/__init__.py':'','rsna/competitive/__init__.py':''}.items():
    p=code/name;assert p.resolve().is_relative_to(code.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);p.chmod(0o600)
sys.path.insert(0,str(code));from rsna.competitive.licensed_parent_population import locate
target_path=locate('/kaggle/input','targets.private.npz',483656,'48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285');license_path=target_path.parent/'source-permission.txt';assert license_path.stat().st_size==4411 and hashlib.sha256(license_path.read_bytes()).hexdigest()=='34c44b20e4d7ace4eb523b0423b36d36dec826fe6ae54c66346796bba0190e79';shutil.copyfile(license_path,root/'source-permission.txt')
if PREVIOUS is not None:
    (root/'fit').mkdir(mode=0o700)
    for key,name in (('checkpoint','checkpoint.pt'),('run','run.json')):
        p=locate('/kaggle/input',name,PREVIOUS[key]['bytes'],PREVIOUS[key]['sha256']);shutil.copyfile(p,root/'fit'/name)
worker=code/'worker.private.py';worker.write_bytes(base64.b64decode(WORKER));worker.chmod(0o600);command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',str(worker)]
if SEGMENT>1:command+=['--resume']
if SEGMENT<3:command+=['--stop-after',str(486*SEGMENT)]
started=time.monotonic()
import threading
finished=threading.Event()
def report_progress():
    last=None
    while not finished.wait(30):
        headers=[json.loads(p.read_text()) for p in sorted(root.glob('allocated-rank-*.json'))]
        p=root/'fit/progress.json';progress=json.loads(p.read_text()) if p.is_file() else None
        if progress!=last or progress is None:print(json.dumps({'segment':SEGMENT,'allocated_headers':headers,'actual_optimizer_progress':progress,'elapsed_seconds':time.monotonic()-started}),flush=True);last=progress
monitor=threading.Thread(target=report_progress,daemon=True);monitor.start()
try:
    with (root/'training.private.log').open('x') as log:result=subprocess.run(command,env={**os.environ,'PYTHONPATH':str(code)},stdout=log,stderr=subprocess.STDOUT,timeout=6.75*3600)
finally:finished.set();monitor.join(timeout=2)

assert result.returncode==0,'Full original fit session failed; preserve exact logs/state and reconcile read-only'
import torch
def sha(p):
    with p.open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()
state=torch.load(root/'fit/checkpoint.pt',map_location='cpu',weights_only=True);run=json.loads((root/'fit/run.json').read_text());assert state['binding']==run['binding'] and all(state['binding'][k]==v for k,v in BINDING.items());assert state['binding']['population']==3887 and state['binding']['recipe']['epochs']==3 and state['binding']['recipe']['windows']==94 and state['binding']['world_size']==2 and state['binding']['total_steps']==1458;assert state['optimizer_boundary'] is True and state['completed']==486*SEGMENT and state['epoch']==SEGMENT and state['cursor']==0 and len(state['history'])==state['completed'] and len(state['rng_by_rank'])==len(state['scaler_by_rank'])==2;assert state['schedule']=={'completed':486*SEGMENT,'total':1458,'kind':'warmup_cosine'} and [r['update'] for r in state['history']]==list(range(1,486*SEGMENT+1));assert all(r['studies']==(7 if r['batch']==486 else 8) for r in state['history']);del state
devices=[json.loads((root/('fit/actual-rank-'+str(i)+'.json')).read_text()) for i in range(2)];assert all(r['GPU']=='Tesla T4' and r['completed']==486*SEGMENT and r['trainable_parameters']==73136096 and r['all_BN_buffers_equal_parent'] is True for r in devices);epoch=root/'fit/epoch-checkpoints'/('epoch-'+str(SEGMENT).zfill(3)+'.pt');assert epoch.is_file();paths={'checkpoint':root/'fit/checkpoint.pt','run':root/'fit/run.json','epoch':epoch};receipt={'status':'actual_whole94_full_parent_epoch_session_complete','segment':SEGMENT,'completed':486*SEGMENT,'total':1458,'worker_sha256':sha(worker),'binding':BINDING,**{k:{'bytes':p.stat().st_size,'sha256':sha(p)} for k,p in paths.items()},'devices':devices,'all_retained_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),'elapsed_seconds':time.monotonic()-started,'fixed_final_complete':SEGMENT==3,'clinical_or_competition_accuracy_claim':False,'competition_entry':False};assert receipt['all_retained_bytes']<4*2**30 and not Path('/kaggle/working/submission.csv').exists();(Path('/kaggle/working')/'licensed-parent-full-session-accepted.json').write_text(json.dumps(receipt,sort_keys=True,indent=2));print(json.dumps(receipt),flush=True)
"""
    for name, value in {
        "FILES_VALUE": files,
        "WORKER_VALUE": base64.b64encode(worker.encode()).decode(),
        "BINDING_VALUE": binding,
        "SEGMENT_VALUE": segment,
        "PREVIOUS_VALUE": previous,
    }.items():
        bootstrap = bootstrap.replace(name, repr(value))
    compile(bootstrap, "private_whole94_checkpoint_continuous_full_session", "exec")
    return bootstrap, {
        "source_sha256": digest(bootstrap),
        "worker_sha256": digest(worker),
        "binding": binding,
        "segment": segment,
        "completed_target": segment * 486,
        "maximum_runtime_seconds": 7 * 3600,
        "GPU_world_size": 2,
        "competition_entry": False,
        "full_production_fit": True,
    }
