"""Extend the actual accepted MRI diagnostic without changing its numerical modules."""

import ast
import base64
import hashlib

ORIGINAL = "a01dd9643355dfb1d2b2f7f43e3cda819e0c44e69a0cd7c2acd99440c0e25554"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def one(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Unique original template marker required")
    return text.replace(old, new)


def assignments(text, replacements):
    spans = []
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            name = node.targets[0]
            if isinstance(name, ast.Name) and name.id in replacements:
                spans.append((node.lineno, node.end_lineno, name.id))
    if len(spans) != len(replacements):
        raise ValueError("Complete immutable source assignments required")
    lines = text.splitlines(keepends=True)
    for start, end, name in sorted(spans, reverse=True):
        lines[start - 1 : end] = [name + "=" + repr(replacements[name]) + "\n"]
    return "".join(lines)


def build(original, sampler, engine, old_receipt):
    if sha(original) != ORIGINAL:
        raise ValueError("Exact actually accepted original MRI diagnostic required")
    if not old_receipt["same_initial_complete_model_and_allrank_RNG_between_roles"]:
        raise ValueError("Original matched MRI admission required")
    values = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(original).body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("FILES", "WORKER")
    }
    if set(values) != {"FILES", "WORKER"} or {"sampling.py", "finding_fit.py"} & set(
        values["FILES"]
    ):
        raise ValueError("Complete original closure without module collisions required")
    worker = base64.b64decode(values["WORKER"], validate=True).decode()
    helper = """
def _finding_components(model,data):
    from rsna.competitive.mil_sampling import readout_moments
    from sampling import components
    names=('norm.weight','norm.bias','att.0.weight','att.0.bias','att.3.weight','att.3.bias','clsW','clsb')
    state=model.state_dict();head={'format':'licensed_parent_MIL_readout_v1','parent_checkpoint_sha256':PARENT_SHA,'source_notebook_sha256':_CACHE_PARENT_SHA,'model':{n:state[n].detach().cpu().float().clone() for n in names}}
    out=[]
    for i in data.indices:
        row=data.frows[i];raw=(data.features/row['file']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:features=torch.from_numpy(arrays['features'].copy())
        weights,votes,bias,_=readout_moments(head,features);out.append(components(weights,votes,bias)[0])
    return np.stack(out)

"""
    for before, after in (
        ("def main():\n", helper + "def main():\n"),
        (
            "from rsna.competitive.licensed_importance_train import fit",
            "from finding_fit import fit",
        ),
        ("choices=('uniform','influence')", "choices=('iid','finding_strata')"),
        (
            "initial_identity=_initial_identity(model,device);",
            "finding_components=_finding_components(model,original_data);initial_identity=_initial_identity(model,device);",
        ),
        (
            "Recipe(epochs=2,role=a.role),device=device",
            "Recipe(epochs=2,role='influence'),components=finding_components,allocation=a.role,device=device",
        ),
    ):
        worker = one(worker, before, after)
    worker = worker.replace(
        "/kaggle/working/importance-recovery", "/kaggle/working/finding-strata-recovery"
    )
    compile(worker, "finding_strata_actual_MRI_worker", "exec")
    files = {**values["FILES"], "sampling.py": sampler, "finding_fit.py": engine}
    source = assignments(
        original, {"FILES": files, "WORKER": base64.b64encode(worker.encode()).decode()}
    )
    source = source.replace(
        "/kaggle/working/importance-recovery", "/kaggle/working/finding-strata-recovery"
    )
    if source.count("for role in ('uniform','influence'):") != 2:
        raise ValueError("Original phase and collector role loops required")
    source = source.replace(
        "for role in ('uniform','influence'):", "for role in ('iid','finding_strata'):"
    )
    marker = "roles={};identities=[];finals=[]"
    if source.count(marker) != 1:
        raise ValueError("Original full state/prediction collector boundary required")
    source = source.split(marker)[0] + COLLECTOR
    # Bind the actual original control digest in the active collector.
    source = one(
        source,
        "ORIGINAL_PREDICTION_LITERAL",
        repr(old_receipt["roles"]["influence"]["prediction_SHA"]),
    )
    compile(source, "finding_strata_actual_MRI_recovery", "exec")
    return source, {
        "source_sha256": sha(source),
        "worker_sha256": sha(worker),
        "original_source_sha256": ORIGINAL,
        "original_numerical_modules_unchanged": all(
            files[n] == text for n, text in values["FILES"].items()
        ),
        "additional_modules_sha256": {"sampling.py": sha(sampler), "finding_fit.py": sha(engine)},
        "original_control_prediction_sha256": old_receipt["roles"]["influence"]["prediction_SHA"],
        "roles": ["iid", "finding_strata"],
        "updates_each_role": 4,
        "new_Dataset_or_pixels": False,
        "production_or_entry_authority": False,
    }


COLLECTOR = r"""
roles={};identities=[];finals=[];complete=[]
for role in ('iid','finding_strata'):
    d=root/role;full=torch.load(d/'full/checkpoint.pt',map_location='cpu',weights_only=True);resumed=torch.load(d/'resumed/checkpoint.pt',map_location='cpu',weights_only=True);same(full,resumed)
    assert full['completed']==4 and full['binding']['world_size']==2 and full['binding']['recipe']['windows']==16 and full['binding']['recipe']['role']=='influence'
    allocation=full['binding']['finding_allocation'];assert allocation['allocation']==role and allocation['mixture_density_unchanged'] is True and allocation['images_per_study']==16
    assert [x['update'] for x in full['history']]==[1,2,3,4]
    for epoch in (1,2):same(torch.load(d/'full/epoch-checkpoints'/('epoch-'+str(epoch).zfill(3)+'.pt'),map_location='cpu',weights_only=True),torch.load(d/'resumed/epoch-checkpoints'/('epoch-'+str(epoch).zfill(3)+'.pt'),map_location='cpu',weights_only=True))
    pred=(d/'full/engineering-predictions.private.npy').read_bytes();assert pred==(d/'resumed/engineering-predictions.private.npy').read_bytes();devices=[json.loads((d/'full'/('full-rank-'+str(rank)+'.json')).read_text()) for rank in (0,1)]
    assert all(x['GPU']=='Tesla T4' and x['finite_updates']==4 and x['trainable_parameters']==73136096 and x['all_BN_buffers_equal_parent'] for x in devices)
    assert all(x['gradient_norm']>0 and x['loss']>=0 for x in full['history'])
    identities.append([x['initial_identity'] for x in devices]);finals.append(full['model']);complete.append(full)
    roles[role]={'complete_model_AdamW_LR_cursors_allrank_RNG_scaler_history_and_epoch_snapshots_equal':True,'all16_full94_two_view_prediction_bytes_equal':True,'prediction_SHA':hashlib.sha256(pred).hexdigest(),'proposal_SHA':full['binding']['teacher_proposal_sha256'],'components_SHA':allocation['components_sha256'],'finite_updates':4,'devices':devices}
same(identities[0],identities[1]);assert roles['iid']['proposal_SHA']==roles['finding_strata']['proposal_SHA'] and roles['iid']['components_SHA']==roles['finding_strata']['components_SHA']
old_receipts=list(Path('/kaggle/input').rglob('licensed-importance-real-MRI-recovery-accepted.json'));assert len(old_receipts)==1
old=old_receipts[0].parent/'importance-recovery/influence/full/checkpoint.pt';prior=torch.load(old,map_location='cpu',weights_only=True)
current=dict(complete[0]);current['binding']=dict(current['binding']);del current['binding']['finding_allocation'];same(current,prior)
assert roles['iid']['prediction_SHA']==ORIGINAL_PREDICTION_LITERAL
distinct=sum(not torch.equal(finals[0][n],value) for n,value in finals[1].items());assert distinct>0
files_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file());assert files_bytes<12*2**30 and not Path('/kaggle/working/submission.csv').exists()
receipt={'status':'actual_finding_strata_bothT4_original_control_and_recovery_accepted','fit_studies':16,'disjoint_engineering_studies':16,'epochs':2,'updates_each_role':4,'sampled_windows':16,'full_inference_windows':94,'roles':roles,'original_influence_complete_control_state_and_prediction_bytes_equal':True,'same_initial_complete_model_and_allrank_RNG_between_roles':True,'distinct_final_learned_tensors':distinct,'source_module_sha256':{n:hashlib.sha256(text.encode()).hexdigest() for n,text in FILES.items()},'worker_source_sha256':hashlib.sha256(base64.b64decode(WORKER)).hexdigest(),'private_target_sha256':hashlib.sha256(base64.b64decode(TARGETS)).hexdigest(),'all_retained_bytes':files_bytes,'elapsed_seconds':time.monotonic()-start,'phases':phases,'clinical_or_competition_accuracy_claim':False,'full_production_fit_dispatched':False,'competition_entry':False}
(Path('/kaggle/working')/'finding-strata-real-MRI-recovery-accepted.json').write_text(json.dumps(receipt,sort_keys=True,indent=2));print(json.dumps(receipt),flush=True)
"""
