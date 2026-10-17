"""Compile fixed3 full-population sampling comparisons after actual GPU recovery.

Original model/renderer source is supplied privately. No external body, fitting
record, dataset mutation or competition submission is embedded in this module.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json

from rsna.competitive.licensed_parent_population import POPULATION_SHA, validate_proofs

RECOVERY_SHA = "a01dd9643355dfb1d2b2f7f43e3cda819e0c44e69a0cd7c2acd99440c0e25554"
WORKER_SHA = "cbe40bfab07a1a24d990dd8b4c8b9f4b3713fde12a200ce7d1d73b28e5557f60"
PROPOSAL_SHA = "e812a9f90cd2b3db92d9acafd28688d5ee6fe16fc188d6517c2bab565f79205e"
ACCEPTED_MODULES = {
    "rsna/competitive/licensed_parent_train.py": "cf238d9154806a51cd92c80a61e39ca344ef865558305db51cbfdc86f89eeec0",
    "rsna/competitive/checkpoint.py": "a70ef871e200bac89e4f071d9394ce8b0abc1c2ba0017eaba3692d25dde31e21",
    "rsna/competitive/released_context.py": "4ed7cd1978a4acac6c233c734914dbe713aed818f229eabd7f23fa10580fbee9",
    "rsna/competitive/licensed_full_bag.py": "74a4e19eb37570f66754fa3b4d8e2e117fadb7cc91c61345efc4e962cbf3983f",
    "rsna/competitive/licensed_importance.py": "fc3fcedfc52fc922b82f7cb8e2034b6d7526d65c536c2ae8163aa8de1d3dafe4",
    "rsna/competitive/licensed_importance_train.py": "e024a8ee91d5d64967e6128bfeac1ab2774468613ccaa43d13fc8ea569a5de7d",
    "rsna/competitive/mil_sampling.py": "f04b4e9ff0874d1809742fed4b9fafc57e77255a9908e2d56c7fdf8203a0a54e",
    "rsna/competitive/released_context_train.py": "5addea8212501ba31021c739543a7fa627cf70066ad439fa0889ba5c5b3fa59f",
    "rsna/competitive/released_context_infer.py": "4dbeb46a3e647a840047e6cede75328a9945a57ec2db4261250d03e28452cc34",
    "rsna/schema.py": "d0775e4c91cc696aa771828c12b5f649edebd81fdb4a626b2c715d62f5394948",
}
POPULATION_MODULES = {
    "rsna/competitive/licensed_pixel_cache.py": "7bdaf41a864b499401ff4a62e6288d30eead3849c0fda5c49cc67b3b2d0e2ab5",
    "rsna/competitive/licensed_pixel_bags.py": "857a374f9f6f88b10028b6a899961d6c62c64151f800710deaa20a915cb5616e",
    "rsna/competitive/licensed_parent_population.py": "66952da8dd25126a71ccd218a91403959a43a6a6d73390e3dc8d7d105a5a706e",
}


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def accepted_recovery(actual):
    r = actual.get("receipt", {})
    if (
        actual.get("status") != "COMPLETE"
        or actual.get("kernel_id") != 139130771
        or actual.get("version") != 1
        or actual.get("source_sha256") != RECOVERY_SHA
        or r.get("status")
        != "actual_matched_uniform_influence_full_encoder_twoT4_MRI_recovery_accepted"
        or r.get("worker_source_sha256") != WORKER_SHA
        or r.get("source_module_sha256") != ACCEPTED_MODULES
        or r.get("fit_studies") != 16
        or r.get("disjoint_engineering_studies") != 16
        or r.get("updates_per_role") != 4
        or r.get("same_initial_complete_model_and_allrank_RNG_between_roles") is not True
        or set(r.get("roles", {})) != {"uniform", "influence"}
        or r.get("distinct_final_learned_tensors", 0) <= 0
    ):
        raise ValueError("Actual exact two-role full encoder GPU recovery required")
    for role in r["roles"].values():
        if (
            role.get(
                "complete_model_AdamW_LR_cursors_allrank_RNG_scaler_history_and_epoch_snapshots_equal"
            )
            is not True
            or role.get("all16_full94_two_view_prediction_bytes_equal") is not True
            or len(role.get("devices", [])) != 2
            or any(
                d.get("GPU") != "Tesla T4"
                or d.get("trainable_parameters") != 73136096
                or d.get("all_BN_buffers_equal_parent") is not True
                for d in role["devices"]
            )
        ):
            raise ValueError("Complete learned state, both T4s and original predictions required")


def proposal_admission(value):
    if (
        value.get("status") != "all3887_fixed_fitting_teacher_influence_proposals_CPU_accepted"
        or value.get("population") != 3887
        or value.get("population_sha256") != POPULATION_SHA
        or value.get("shape") != [3887, 94]
        or value.get("dtype") != "float64"
        or value.get("uniform_fraction") != 0.25
        or value.get("proposal_table") != {"bytes": 2923152, "sha256": PROPOSAL_SHA}
        or value.get("proposal_module_sha256")
        != ACCEPTED_MODULES["rsna/competitive/mil_sampling.py"]
        or value.get("clinical_labels_reports_Gold_confirmation_read") is not False
        or value.get("new_CNN_GPU_or_optimizer") is not False
    ):
        raise ValueError("Exact complete fitting-only fixed teacher proposal table required")


def build(
    accepted_source, actual, files, proofs, proposal, source_git, role, segment, *, previous=None
):
    accepted_recovery(actual)
    proposal_admission(proposal)
    validate_proofs(proofs)
    if (
        digest(accepted_source) != RECOVERY_SHA
        or set(files) != {*ACCEPTED_MODULES, *POPULATION_MODULES}
        or any(digest(files[n]) != h for n, h in {**ACCEPTED_MODULES, **POPULATION_MODULES}.items())
        or role not in ("uniform", "influence")
        or type(segment) is not int
        or not 1 <= segment <= 3
        or not isinstance(source_git, str)
        or len(source_git) != 40
        or any(c not in "0123456789abcdef" for c in source_git)
    ):
        raise ValueError(
            "Exact accepted source, immutable numerical closure and fixed role/epoch required"
        )
    literals = {}
    for node in ast.parse(accepted_source).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("WORKER", "PERMISSION")
        ):
            literals[node.targets[0].id] = ast.literal_eval(node.value)
    if set(literals) != {"WORKER", "PERMISSION"}:
        raise ValueError("Original private model worker and source permission required")
    worker = base64.b64decode(literals["WORKER"], validate=True).decode()
    if digest(worker) != WORKER_SHA:
        raise ValueError("Actual accepted numerical worker required")
    binding = {
        "production_plan": "licensed_MRI_parent_uniform_influence_IID16_full3887_fixed3_continuous_v1",
        "source_git": source_git,
        "role": role,
        "parent_checkpoint_sha256": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef",
        "parent_source_sha256": "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde",
        "source_module_sha256": {n: digest(s) for n, s in sorted(files.items())},
        "actual_recovery": {
            "kernel_id": 139130771,
            "source_sha256": RECOVERY_SHA,
            "worker_sha256": WORKER_SHA,
            "receipt_sha256": digest(
                json.dumps(actual["receipt"], sort_keys=True, separators=(",", ":"))
            ),
        },
        "proposal": proposal,
        "pixel_collection_proofs_sha256": digest(
            json.dumps(proofs, sort_keys=True, separators=(",", ":"))
        ),
        "original_target_sha256": "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285",
        "training_bag": "IID16_with_replacement_centered_inverse_proposal_checkpointed8",
        "diagnostic_only": False,
        "independent_internal_validation": False,
        "protected_gold_confirmation_used": False,
        "clinical_or_competition_accuracy_claim": False,
    }
    main = (
        r"""
from rsna.competitive.licensed_parent_population import load as load_population,locate
_PROOFS=PROOFS_VALUE
_BINDING=BINDING_VALUE
_ROLE=ROLE_VALUE

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');parser.add_argument('--stop-after',type=int);a=parser.parse_args();assert a.stop_after in (None,486,972);rank=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==2;torch.cuda.set_device(rank);device=torch.device('cuda',rank);assert torch.cuda.get_device_name(device)=='Tesla T4';torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True);random.seed(20261016+rank);np.random.seed(20261016+rank);torch.manual_seed(20261016+rank);torch.cuda.manual_seed_all(20261016+rank);dist.init_process_group('nccl');root=Path('/kaggle/working/importance-full');render=lambda volume,mask,mirror:eval_windows(volume,mask,94,320,norm=NORM,mirror=mirror);original=load_population('/kaggle/input',_PROOFS,render);assert len(original)==3887;p=locate('/kaggle/input','proposals.private.npy',2923152,_BINDING['proposal']['proposal_table']['sha256']);q=np.load(p,allow_pickle=False);assert q.shape==(3887,94) and q.dtype==np.float64;data=ProposalCases(original,torch.from_numpy(q));del q
    weight=next(p for directory in Path('/kaggle/input').iterdir() for p in directory.rglob('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt') if p.is_file());base,res=load_model(str(weight),'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',384,device);model=_whole_model(base);del base;assert res==320 and all(p.requires_grad for p in model.parameters()) and sum(p.numel() for p in model.parameters())==73136096;output=root/'fit';initial_bn={n:v.detach().cpu().clone() for n,v in model.state_dict().items() if any(k in n for k in ('running_mean','running_var','num_batches_tracked'))};initial=_initial_identity(model,device);output.mkdir(parents=True,exist_ok=True);(root/('allocated-rank-'+str(rank)+'.json')).write_text(json.dumps({'rank':rank,'GPU':torch.cuda.get_device_name(device),'studies':3887,'role':_ROLE,'windows':16,'trainable_parameters':73136096,'proposal_SHA':data.proposal_sha256,'original_inputs_accepted':True,'initial_identity':initial,'resume_requested':a.resume}));start=time.monotonic();result=fit(model,data,output,_BINDING,Recipe(epochs=3,role=_ROLE),device=device,resume=a.resume,stop_after=a.stop_after);fit_seconds=time.monotonic()-start
    assert result['total']==1458 and result['completed'] in (486,972,1458) and all(torch.equal(v,model.state_dict()[n].detach().cpu()) for n,v in initial_bn.items());info={'rank':rank,'GPU':torch.cuda.get_device_name(device),'total_VRAM_bytes':torch.cuda.get_device_properties(device).total_memory,'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'completed':result['completed'],'total':1458,'fit_seconds':fit_seconds,'all_BN_buffers_equal_parent':True,'trainable_parameters':73136096,'CPU_threads_per_rank':torch.get_num_threads(),'role':_ROLE,'initial_identity':initial,'clinical_or_competition_accuracy_claim':False};(output/('actual-rank-'+str(rank)+'.json')).write_text(json.dumps(info,sort_keys=True));dist.barrier();dist.destroy_process_group()
""".replace("PROOFS_VALUE", repr(proofs))
        .replace("BINDING_VALUE", repr(binding))
        .replace("ROLE_VALUE", repr(role))
    )
    nodes = [
        n for n in ast.parse(worker).body if isinstance(n, ast.FunctionDef) and n.name == "main"
    ]
    if len(nodes) != 1:
        raise ValueError("One unchanged real trainer worker entry point required")
    lines = worker.splitlines(keepends=True)
    node = nodes[0]
    worker = "".join(lines[: node.lineno - 1]) + main + "\n" + "".join(lines[node.end_lineno :])
    compile(worker, "private_matched_IID16_full3887_worker", "exec")
    if segment == 1:
        if previous is not None:
            raise ValueError("First epoch starts from the same exact MRI parent only")
    elif (
        not isinstance(previous, dict)
        or previous.get("status") != "actual_matched_IID16_full_parent_epoch_session_complete"
        or previous.get("role") != role
        or previous.get("segment") != segment - 1
        or previous.get("completed") != 486 * (segment - 1)
        or previous.get("worker_sha256") != digest(worker)
        or previous.get("binding") != binding
    ):
        raise ValueError("Exact previous same-role optimizer/RNG/source/epoch boundary required")
    if previous is not None:
        for name in ("checkpoint", "run"):
            v = previous.get(name, {})
            if (
                type(v.get("bytes")) is not int
                or not 0 < v["bytes"] <= 2 * 2**30
                or not isinstance(v.get("sha256"), str)
                or len(v["sha256"]) != 64
            ):
                raise ValueError("Original whole checkpoint and run size/SHA required")
    bootstrap = r"""
import os,sys,json,base64,subprocess,time,hashlib,shutil,threading
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
FILES=FILES_VALUE
WORKER=WORKER_VALUE
BINDING=BINDING_VALUE
SEGMENT=SEGMENT_VALUE
ROLE=ROLE_VALUE
PREVIOUS=PREVIOUS_VALUE
PERMISSION=PERMISSION_VALUE
root=Path('/kaggle/working/importance-full');assert not root.exists() and shutil.disk_usage('/kaggle/working').free>6*2**30;root.mkdir(mode=0o700);(root/'source-permission.txt').write_bytes(base64.b64decode(PERMISSION));assert hashlib.sha256((root/'source-permission.txt').read_bytes()).hexdigest()=='34c44b20e4d7ace4eb523b0423b36d36dec826fe6ae54c66346796bba0190e79';code=root/'code';code.mkdir(mode=0o700)
for name,text in {**FILES,'rsna/__init__.py':'','rsna/competitive/__init__.py':''}.items():
    p=code/name;assert p.resolve().is_relative_to(code.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text);p.chmod(0o600)
sys.path.insert(0,str(code));from rsna.competitive.licensed_parent_population import locate
weights=[p for d in Path('/kaggle/input').iterdir() for p in d.rglob('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt') if p.is_file()];assert len(weights)==1
with weights[0].open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==BINDING['parent_checkpoint_sha256']
if PREVIOUS is not None:
    (root/'fit').mkdir(mode=0o700)
    for key,name in (('checkpoint','checkpoint.pt'),('run','run.json')):
        p=locate('/kaggle/input',name,PREVIOUS[key]['bytes'],PREVIOUS[key]['sha256']);shutil.copyfile(p,root/'fit'/name)
worker=code/'worker.private.py';worker.write_bytes(base64.b64decode(WORKER));worker.chmod(0o600);command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',str(worker)]
if SEGMENT>1:command+=['--resume']
if SEGMENT<3:command+=['--stop-after',str(486*SEGMENT)]
started=time.monotonic();finished=threading.Event()
def monitor():
    last=None
    while not finished.wait(30):
        headers=[json.loads(p.read_text()) for p in sorted(root.glob('allocated-rank-*.json'))];p=root/'fit/progress.json';progress=json.loads(p.read_text()) if p.is_file() else None
        if progress!=last or progress is None:print(json.dumps({'role':ROLE,'segment':SEGMENT,'allocated_headers':headers,'actual_optimizer_progress':progress,'elapsed_seconds':time.monotonic()-started}),flush=True);last=progress
watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
try:
    with (root/'training.private.log').open('x') as log:result=subprocess.run(command,env={**os.environ,'PYTHONPATH':str(code)},stdout=log,stderr=subprocess.STDOUT,timeout=90*60)
finally:finished.set();watcher.join(timeout=2)
assert result.returncode==0,'Exact full-comparison epoch failed; preserve original state without restart'
import torch
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
state=torch.load(root/'fit/checkpoint.pt',map_location='cpu',weights_only=True);run=json.loads((root/'fit/run.json').read_text());assert state['binding']==run['binding'] and all(state['binding'][k]==v for k,v in BINDING.items());assert state['binding']['recipe']['role']==ROLE and state['binding']['recipe']['windows']==16 and state['binding']['recipe']['epochs']==3 and state['binding']['population']==3887 and state['binding']['population_sha256']=='0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42' and state['binding']['total_steps']==1458;assert state['optimizer_boundary'] is True and state['completed']==486*SEGMENT and state['epoch']==SEGMENT and state['cursor']==0 and len(state['history'])==486*SEGMENT and len(state['rng_by_rank'])==len(state['scaler_by_rank'])==2;assert [x['update'] for x in state['history']]==list(range(1,486*SEGMENT+1)) and all(x['studies']==(7 if x['batch']==486 else 8) for x in state['history']);del state
devices=[json.loads((root/('fit/actual-rank-'+str(i)+'.json')).read_text()) for i in range(2)];assert all(x['GPU']=='Tesla T4' and x['completed']==486*SEGMENT and x['all_BN_buffers_equal_parent'] for x in devices);epoch=root/'fit/epoch-checkpoints'/('epoch-'+str(SEGMENT).zfill(3)+'.pt');assert epoch.is_file();files={'checkpoint':root/'fit/checkpoint.pt','run':root/'fit/run.json','epoch':epoch};receipt={'status':'actual_matched_IID16_full_parent_epoch_session_complete','role':ROLE,'segment':SEGMENT,'completed':486*SEGMENT,'total':1458,'worker_sha256':sha(worker),'binding':BINDING,**{k:{'bytes':p.stat().st_size,'sha256':sha(p)} for k,p in files.items()},'devices':devices,'all_retained_bytes':sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),'elapsed_seconds':time.monotonic()-started,'fixed_final_complete':SEGMENT==3,'clinical_or_competition_accuracy_claim':False,'competition_entry':False};assert receipt['all_retained_bytes']<4*2**30 and not Path('/kaggle/working/submission.csv').exists();(Path('/kaggle/working')/'licensed-importance-full-session-accepted.json').write_text(json.dumps(receipt,sort_keys=True,indent=2));print(json.dumps(receipt),flush=True)
"""
    for name, value in {
        "FILES_VALUE": files,
        "WORKER_VALUE": base64.b64encode(worker.encode()).decode(),
        "BINDING_VALUE": binding,
        "SEGMENT_VALUE": segment,
        "ROLE_VALUE": role,
        "PREVIOUS_VALUE": previous,
        "PERMISSION_VALUE": literals["PERMISSION"],
    }.items():
        bootstrap = bootstrap.replace(name, repr(value))
    compile(bootstrap, "private_full3887_matched_IID16_epoch", "exec")
    return bootstrap, {
        "source_sha256": digest(bootstrap),
        "worker_sha256": digest(worker),
        "binding": binding,
        "role": role,
        "segment": segment,
        "completed_target": segment * 486,
        "maximum_GPU_seconds": 100 * 60,
        "competition_entries": 0,
        "full_production_fit": True,
    }
