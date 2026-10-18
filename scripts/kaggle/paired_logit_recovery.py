"""Compile a NEW paired-correction encoder/FP16/dropout recovery, never a submission."""

from __future__ import annotations

import ast
import base64
import hashlib

from scripts.kaggle import licensed_importance_recovery as original

EXTRA = (
    "rsna/competitive/paired_logit_cases.py",
    "rsna/competitive/paired_logit_control.py",
    "rsna/competitive/paired_logit_train.py",
)
STATUS = "actual_paired_logit_encoder_FP16_dropout_recovery_accepted"


def literals(source):
    return {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("FILES", "WORKER", "TARGETS", "PERMISSION")
    }


def build(parent, codec, files, private_targets, *, permission):
    expected = {*original.MODULES, original.WHOLE_MODULE, *original.EXTRA_MODULES, *EXTRA}
    if set(files) != expected:
        raise ValueError("Complete new corrected-encoder source closure required")
    source, binding = original.build(
        parent,
        codec,
        {k: v for k, v in files.items() if k not in EXTRA},
        private_targets,
        permission=permission,
    )
    values = literals(source)
    worker = base64.b64decode(values["WORKER"], validate=True).decode()

    def change(old, new):
        nonlocal worker
        if worker.count(old) != 1:
            raise ValueError("Original recovery template changed; explicit review required")
        worker = worker.replace(old, new)

    change(
        "class WholeStudyRaptor(ImportanceReadout,CheckpointedEncode,RaptorClassifier):pass",
        "from rsna.competitive.paired_logit_control import PairedLogitReadout\n"
        "class WholeStudyRaptor(PairedLogitReadout,ImportanceReadout,CheckpointedEncode,RaptorClassifier):pass",
    )
    change(
        "from rsna.competitive.licensed_importance_train import fit",
        "from rsna.competitive.paired_logit_train import fit",
    )
    helper = r"""
def _paired_cases(model,data,recipe,root,rank,mode):
    import copy
    from rsna.competitive.paired_logit_control import schedule
    from rsna.competitive.paired_logit_cases import PairedCases
    norm=copy.deepcopy(model.norm).cpu().float().eval();att=copy.deepcopy(model.att).cpu().float().eval()
    weight=model.clsW.detach().cpu().float();bias=model.clsb.detach().cpu().float()
    def parent_readout(features,logq):
        h=norm(features);scores=att(h).float()
        if logq is not None:scores=scores-logq[None,:,None]
        pooled=torch.einsum('bkn,bkf->bnf',scores.softmax(1),h)
        return (pooled*weight).sum(-1)+bias
    entries=[]
    for i in range(len(data)):
        row=data.original.frows[data.original.indices[i]]
        raw=(data.original.features/row['file']).read_bytes()
        assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
        with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:features=torch.from_numpy(arrays['features'].copy())
        entries.append(schedule(parent_readout,features,data.uids[i],recipe,data.proposal(i)))
    arrays={'study_keys':np.array([list(bytes.fromhex(e['study_uid_sha256'])) for e in entries],np.uint8)}
    for name in ('positions','mirror','log_proposal','shift'):arrays[name]=np.stack([e[name].numpy() for e in entries])
    table=root/('new-corrections-'+mode+'-'+str(rank)+'.private.npz')
    with table.open('xb') as stream:np.savez(stream,**arrays)
    assert table.stat().st_size<16384
    return PairedCases(data,table,hashlib.sha256(table.read_bytes()).hexdigest(),recipe)

"""
    change("def main():\n", helper + "def main():\n")
    change(
        "initial_identity=_initial_identity(model,device);engineering=",
        "initial_identity=_initial_identity(model,device);data=_paired_cases(model,data,Recipe(role='influence'),root,rank,a.mode);engineering=",
    )
    change("Recipe(epochs=2,role=a.role)", "Recipe(role=a.role)")
    change("choices=('uniform','influence')", "choices=('influence',)")
    # Replace only the bootstrap AST literals; the executed worker body is encoded.
    old_worker = repr(values["WORKER"])
    source = source.replace(old_worker, repr(base64.b64encode(worker.encode()).decode()), 1)
    source = source.replace(repr(values["FILES"]), repr(files), 1)
    source = source.replace("for role in ('uniform','influence'):", "for role in ('influence',):")
    source = source.replace("full['completed']==4", "full['completed']==6")
    source = source.replace("for epoch in (1,2):", "for epoch in (1,2,3):")
    source = source.replace("list(range(1,5))", "list(range(1,7))")
    source = source.replace("[1,2,3,4]", "[1,2,3,4,5,6]")
    source = source.replace("x['finite_updates']==4", "x['finite_updates']==6")
    source = source.replace("'finite_updates':4", "'finite_updates':6")
    old = "same(identities[0],identities[1]);assert roles['uniform']['proposal_SHA']==roles['influence']['proposal_SHA'];distinct=sum(not torch.equal(finals[0][n],v) for n,v in finals[1].items());assert distinct>0,'Actual proposal must cause distinct learned tensors to justify full comparison';"
    if source.count(old) != 1:
        raise ValueError("Original cross-role acceptance changed")
    source = source.replace(old, "distinct=None;")
    source = source.replace("'same_initial_complete_model_and_allrank_RNG_between_roles':True,", "")
    source = source.replace("'epochs':2,'updates_per_role':4", "'epochs':3,'updates_per_role':6")
    source = source.replace(
        "actual_matched_uniform_influence_full_encoder_twoT4_MRI_recovery_accepted", STATUS
    )
    source = source.replace(
        "licensed-importance-real-MRI-recovery-accepted.json",
        "paired-logit-real-MRI-recovery-accepted.json",
    )
    source = source.replace("importance-recovery", "paired-logit-recovery")
    compile(
        worker.replace("importance-recovery", "paired-logit-recovery"), "new_paired_worker", "exec"
    )
    # The worker is encoded: update its root separately, not just outer text.
    encoded_worker = base64.b64encode(worker.encode()).decode()
    worker = worker.replace("importance-recovery", "paired-logit-recovery")
    source = source.replace(
        repr(encoded_worker), repr(base64.b64encode(worker.encode()).decode()), 1
    )
    compile(source, "new_paired_encoder_recovery", "exec")
    return source, {
        **binding,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "worker_sha256": hashlib.sha256(worker.encode()).hexdigest(),
        "own_module_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in files.items()},
        "roles": ["influence"],
        "epochs": 3,
        "updates": 6,
        "diagnostic_only": True,
        "accuracy_or_production_fit_admitted": False,
    }
