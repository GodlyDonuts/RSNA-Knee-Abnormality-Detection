"""Prepare one combined noise and matched update-cost experiment; no dispatch."""

from __future__ import annotations

import ast
import base64
import hashlib

from scripts.kaggle import paired_encoder_noise_probe as noise

EXTRA = (*noise.EXTRA, "rsna/competitive/paired_step_timing.py")
STATUS = "actual_encoder_sampling_noise_and_matched_update_cost_measured"


def literal(source, name, value=None, *, replace=False):
    nodes = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id == name
    ]
    if len(nodes) != 1:
        raise ValueError("One explicit source literal required: " + name)
    node = nodes[0]
    if not replace:
        return ast.literal_eval(node.value)
    raw = source.encode()
    lines = raw.splitlines(keepends=True)
    assert node.end_lineno is not None and node.end_col_offset is not None
    start = sum(map(len, lines[: node.lineno - 1])) + node.col_offset
    end = sum(map(len, lines[: node.end_lineno - 1])) + node.end_col_offset
    return (raw[:start] + (name + "=" + repr(value)).encode() + raw[end:]).decode()


def build(recovery_source, files, training, publication):
    if set(files) != set(EXTRA):
        raise ValueError("Complete new noise and update-timing source closure required")
    source, old = noise.build(
        recovery_source, {n: files[n] for n in noise.EXTRA}, training, publication
    )
    closure = {
        **literal(source, "FILES"),
        "rsna/competitive/paired_step_timing.py": files["rsna/competitive/paired_step_timing.py"],
    }
    worker = (
        base64.b64decode(literal(source, "WORKER"), validate=True)
        .decode()
        .replace("paired-encoder-noise-v1", "paired-encoder-noise-v2")
    )
    original_worker = base64.b64decode(literal(recovery_source, "WORKER"), validate=True).decode()
    nodes = [
        n
        for n in ast.parse(original_worker).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    ]
    if len(nodes) != 1:
        raise ValueError("One source-bound original worker entry required")
    prefix = "".join(original_worker.splitlines(keepends=True)[: nodes[0].lineno - 1])
    benchmark = (
        prefix
        + r"""

class OrdinaryTimingRaptor(ImportanceReadout,CheckpointedEncode,RaptorClassifier):pass

def main():
    from rsna.competitive import licensed_importance_train as ordinary_engine,paired_logit_train as corrected_engine
    from rsna.competitive.paired_step_timing import timed_fit
    arm=sys.argv[1];run=int(sys.argv[2]);assert arm in ('ordinary','controlled') and run in range(4)
    rank=int(os.environ['LOCAL_RANK']);assert int(os.environ['WORLD_SIZE'])==2 and rank in (0,1)
    torch.cuda.set_device(rank);device=torch.device('cuda',rank);assert torch.cuda.get_device_name(device)=='Tesla T4'
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    random.seed(20261016+rank);np.random.seed(20261016+rank);torch.manual_seed(20261016+rank);torch.cuda.manual_seed_all(20261016+rank)
    root=Path('/kaggle/working/paired-encoder-noise-v2');targets=json.loads((root/'targets.private.json').read_text())
    weight=next(p for d in Path('/kaggle/input').iterdir() for p in d.rglob('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt') if p.is_file());base,res=load_model(str(weight),'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',384,device);assert res==320
    kind=OrdinaryTimingRaptor if arm=='ordinary' else WholeStudyRaptor;model=kind(base.backbone,F_dim=base.backbone.num_features).to(device);model.load_state_dict(base.state_dict(),strict=True);assert model.state_dict().keys()==base.state_dict().keys() and all(torch.equal(t,base.state_dict()[n]) for n,t in model.state_dict().items());del base
    assert sum(p.numel() for p in model.parameters())==73136096 and all(p.requires_grad for p in model.parameters())
    original=PixelCases(targets,list(range(16)));data=ProposalCases(original,_teacher_proposals(model,original));recipe=Recipe(role='influence')
    if arm=='controlled':data=_paired_cases(model,data,recipe,root,rank,'timing-'+str(run))
    identity=_initial_identity(model,device);dist.init_process_group('nccl');torch.cuda.reset_peak_memory_stats(device)
    output=Path('/tmp/rsna-paired-update-timing-v2')/('run-'+str(run));binding={'parent_checkpoint_sha256':PARENT_SHA,'diagnostic_only':True,'experiment':'new_matched_optimizer_cost_only','no_clinical_or_production_acceptance':True}
    measured=timed_fit(ordinary_engine if arm=='ordinary' else corrected_engine,model,data,output,binding,recipe,device=device)
    result={'arm':arm,'run':run,'rank':rank,'initial_identity':identity,'fit_studies':len(data),'parameters':73136096,'GPU':torch.cuda.get_device_name(device),'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),'total_VRAM_bytes':torch.cuda.get_device_properties(device).total_memory,**measured};assert result['actual_optimizer_updates']==6 and 0<result['peak_allocated_bytes']<result['total_VRAM_bytes'];(root/('timing-'+str(run)+'-rank-'+str(rank)+'.json')).write_text(json.dumps(result));dist.barrier();dist.destroy_process_group()

if __name__=='__main__':main()
"""
    )
    compile(worker, "new_noise_v2_worker", "exec")
    compile(benchmark, "new_fresh_process_timing_worker", "exec")
    for name, value in {
        "FILES": closure,
        "WORKER": base64.b64encode(worker.encode()).decode(),
    }.items():
        source = literal(source, name, value, replace=True)
    source = source.replace("paired-encoder-noise-v1", "paired-encoder-noise-v2")
    allocation = "with (root/'new-noise-probe.private.log').open('x') as log:"
    if source.count(allocation) != 1:
        raise ValueError("One initial measurement allocation boundary required")
    parent_check = """parent_weights=[p for d in Path('/kaggle/input').iterdir() for p in d.rglob('raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt') if p.is_file()];assert len(parent_weights)==1
with parent_weights[0].open('rb') as parent_stream:assert hashlib.file_digest(parent_stream,'sha256').hexdigest()=='7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
"""
    source = source.replace(allocation, parent_check + allocation, 1)
    anchor = (
        "rows=[json.loads((root/('case-'+str(i)+'.private.json')).read_text()) for i in range(4)]"
    )
    if source.count(anchor) != 1:
        raise ValueError("One completed noise-measurement boundary required")
    code = r"""
BENCHMARK=BENCHMARK_VALUE
bench=code/'timing-worker.private.py';bench.write_bytes(base64.b64decode(BENCHMARK));bench.chmod(0o600)
temporary=Path('/tmp/rsna-paired-update-timing-v2');assert not temporary.exists() and shutil.disk_usage('/tmp').free>12*2**30;temporary.mkdir(mode=0o700)
timing_records=[];absolute_deadline=start+2550
for run,arm in enumerate(('ordinary','controlled','controlled','ordinary')):
    remaining=absolute_deadline-time.monotonic()-30;assert remaining>0,'Combined experiment deadline exhausted; no new retry'
    with (root/('new-update-timing-'+str(run)+'.private.log')).open('x') as log:
        timed=subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',str(bench),arm,str(run)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=remaining)
    assert timed.returncode==0,'New actual optimizer-cost comparison failed; retain original logs, no automatic retry'
    pair=[json.loads((root/('timing-'+str(run)+'-rank-'+str(rank)+'.json')).read_text()) for rank in (0,1)];assert all(x['arm']==arm and x['run']==run and x['fit_studies']==16 and x['actual_optimizer_updates']==6 for x in pair);assert len(pair[0]['step_timings'])==6 and pair[1]['step_timings']==[];timing_records.append(pair)
assert all(timing_records[run][rank]['initial_identity']==timing_records[0][rank]['initial_identity'] for run in range(4) for rank in (0,1)), 'Both arms/repetitions need identical initial model and all-rank RNG'
temporary_bytes=sum(p.stat().st_size for p in temporary.rglob('*') if p.is_file());assert temporary_bytes<8*2**30
costs={arm:[s['seconds'] for pair in timing_records if pair[0]['arm']==arm for s in pair[0]['step_timings']] for arm in ('ordinary','controlled')};warm_costs={arm:[s['seconds'] for pair in timing_records if pair[0]['arm']==arm for s in pair[0]['step_timings'][1:]] for arm in ('ordinary','controlled')}
assert all(len(x)==12 and all(t>0 for t in x) for x in costs.values());timing_summary={'fresh_two_rank_processes_per_run':True,'order':['ordinary','controlled','controlled','ordinary'],'fixed_parent_initialization_and_all_rank_RNG_equal':True,'runs':timing_records,'step_seconds':costs,'step_sum_ratio':sum(costs['controlled'])/sum(costs['ordinary']),'step_sum_ratio_excluding_first_each_run':sum(warm_costs['controlled'])/sum(warm_costs['ordinary']),'temporary_checkpoint_bytes':temporary_bytes,'temporary_checkpoints_uploaded_or_published':False,'retained_on_VM_only_until_teardown':True,'total_actual_optimizer_updates_all_runs':24,'production_or_accuracy_accepted':False}
""".replace("BENCHMARK_VALUE", repr(base64.b64encode(benchmark.encode()).decode()))
    source = source.replace(anchor, code + "\n" + anchor, 1)
    write = "(Path('/kaggle/working')/'paired-encoder-noise-measured.json').write_text"
    if source.count(write) != 1:
        raise ValueError("One new combined outcome receipt required")
    source = source.replace(
        write,
        "receipt.update(status="
        + repr(STATUS)
        + ",actual_update_timing=timing_summary,benchmark_worker_sha256="
        + repr(hashlib.sha256(benchmark.encode()).hexdigest())
        + ");"
        + write,
        1,
    )
    if literal(source, "FILES") != closure:
        raise ValueError("Wrapper preparation must not alter any numerical module")
    compile(source, "new_combined_encoder_noise_update_cost", "exec")
    return source, {
        **old,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "worker_sha256": hashlib.sha256(worker.encode()).hexdigest(),
        "benchmark_worker_sha256": hashlib.sha256(benchmark.encode()).hexdigest(),
        "own_module_sha256": {
            n: hashlib.sha256(t.encode()).hexdigest() for n, t in closure.items()
        },
        "GPU_session_cap_seconds": 2700,
        "benchmark_updates_total": 24,
        "temporary_VM_checkpoint_cap_bytes": 8 * 2**30,
        "minimum_temporary_disk_free_bytes": 12 * 2**30,
        "temporary_checkpoints_published": False,
        "prepared_only": True,
        "production_or_accuracy_admitted": False,
    }
