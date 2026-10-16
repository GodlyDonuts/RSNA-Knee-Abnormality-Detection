"""Serve a fully verified whole94 epoch through the exact licensed image graph."""

from __future__ import annotations

import ast
import hashlib

from rsna.competitive.licensed_parent_serving import provenance
from scripts.kaggle.series_boundary_notebook import GUARD, PARENT_SOURCE_SHA256, WEIGHTS_SHA256

SERVING_SHA = "f066a6fc53e3b44edaae57bb25637869b875f90cdfb49884a8abcb1413acd804"
BOUNDED_SHA = "5576dff1348ebb7fa322c0ce78f7fdf8aa4b9ef1f2a6be2b757765fcc733cd42"
CODEC_SHA = "7bdaf41a864b499401ff4a62e6288d30eead3849c0fda5c49cc67b3b2d0e2ab5"
MIXIN_SHA = "74a4e19eb37570f66754fa3b4d8e2e117fadb7cc91c61345efc4e962cbf3983f"


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def build(parent, bounded, serving, codec, mixin, training, publication, *, mode):
    model_sha = training.get("receipt", {}).get("epoch", {}).get("sha256")
    epoch = provenance(training, publication, model_sha)
    if (
        mode not in ("qualification", "competition")
        or digest(parent) != PARENT_SOURCE_SHA256
        or digest(serving) != SERVING_SHA
        or digest(bounded) != BOUNDED_SHA
        or digest(codec) != CODEC_SHA
        or digest(mixin) != MIXIN_SHA
        or parent.count(GUARD) != 1
        or parent.count("# %% [code]") < 1
        or parent[parent.index(GUARD) + len(GUARD) :].strip()
    ):
        raise ValueError(
            "Exact licensed image source, own admitted helpers and known serving mode required"
        )
    compile(serving, "strict_full_parent_epoch_serving", "exec")
    source = parent[parent.index("# %% [code]") : parent.index(GUARD)]
    arms = [
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "ARMS" for t in n.targets)
    ]
    if len(arms) != 1:
        raise ValueError("One original single-model arm required")
    old_arm = ast.literal_eval(arms[0].value)
    if (
        len(old_arm) != 1
        or old_arm[0]["file"] != "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
    ):
        raise ValueError("Original single full MRI model required")
    arm = [{**old_arm[0], "file": f"epoch-{epoch:03d}.pt"}]
    lines = source.splitlines(keepends=True)
    source = (
        "".join(lines[: arms[0].lineno - 1])
        + "ARMS="
        + repr(arm)
        + "\n"
        + "".join(lines[arms[0].end_lineno :])
    )
    replacements = {
        WEIGHTS_SHA256: model_sha,
        'torch.load(pt_path, map_location="cpu", weights_only=False)': 'torch.load(pt_path, map_location="cpu", weights_only=True)',
        "nproc = max(4, min(96, (os.cpu_count() or 8) - 8))": "nproc = 2",
        "pool.imap_unordered(_pre_one, test_ids, chunksize=2)": "bounded_results(pool, _pre_one, test_ids, limit=4)",
        "q = queue.Queue(maxsize=QUEUE_MAX)": "q = queue.Queue(maxsize=2)",
        'print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)': 'print(json.dumps({"adapted_MRI_fallback_index":i,"error_type":type(e).__name__}),flush=True)',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("Unique accepted weight/loader/bounded-memory sites required")
        source = source.replace(old, new)
    patch = r"""
from pathlib import Path
_ADAPTED_MODE=MODE_VALUE
_ADAPTED_TRAINING=TRAINING_VALUE
_ADAPTED_PUBLICATION=PUBLICATION_VALUE
_ADAPTED_SHA=MODEL_SHA_VALUE
_ADAPTED_HELPER=SERVING_VALUE
_ADAPTED_CODEC=CODEC_VALUE
_ADAPTED_MIXIN=MIXIN_VALUE
_adapted_counts={'started':0,'completed':0,'failed':0,'decode_errors':0,'empty_views':0}
_adapted_lock=threading.Lock()
_adapted_qualifying=[True]
_adapted_original_infer=infer_probs
_adapted_original_windows=gpu_windows

def _adapted_locate(name,size,digest):
    matches=[]
    for directory in Path('/kaggle/input').iterdir():
        if not directory.is_dir():continue
        for root,dirs,files in os.walk(directory,followlinks=False):
            if any(n in files or n in dirs for n in ('train.csv','test.csv','train_series','train_images')):dirs[:]=[];continue
            if len(Path(root).relative_to(directory).parts)>=5:dirs[:]=[]
            if name in files:
                p=Path(root)/name
                if not p.is_symlink() and p.stat().st_size==size:
                    with p.open('rb') as f:observed=hashlib.file_digest(f,'sha256').hexdigest()
                    if observed==digest:matches.append(p)
    assert len(matches)==1,'One exact immutable original model/input required'
    return matches[0]

def find_weight_file(name):
    assert name==ARMS[0]['file']
    return str(_adapted_model_path)

def infer_probs(model,xwins,device):
    with _adapted_lock:_adapted_counts['started']+=1
    try:result=_adapted_original_infer(model,xwins,device)
    except Exception:
        with _adapted_lock:_adapted_counts['failed']+=1
        raise
    with _adapted_lock:_adapted_counts['completed']+=1
    return result

def gpu_windows(vol,mask,k,res,dev,norm=NORM,mirror=False):
    if not _adapted_qualifying[0] and not mask.any():
        with _adapted_lock:_adapted_counts['empty_views']+=1
    return _adapted_original_windows(vol,mask,k,res,dev,norm=norm,mirror=mirror)

def _pre_one(sid):
    try:vol,mask=_READER(sid,_SER,_TSDIR)
    except Exception as exc:
        with _adapted_lock:_adapted_counts['decode_errors']+=1
        print(json.dumps({'adapted_preprocessing_error_type':type(exc).__name__}),flush=True);vol=np.zeros((MAXS,IMG,IMG),np.uint8);mask=np.zeros(MAXS,np.uint8)
    return sid,vol,mask

def _adapted_boot():
    global _adapted_model_path
    assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert K_EVAL==94 and INPUT_RES==320 and SLOT_SIZES==[26,22,18,12,18] and len(ARMS)==1
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    _adapted_model_path=_adapted_locate(ARMS[0]['file'],_ADAPTED_TRAINING['receipt']['epoch']['bytes'],_ADAPTED_SHA)
    snapshot=torch.load(_adapted_model_path,map_location='cpu',weights_only=True);ns={'__name__':'strict_adapted_epoch_serving'};exec(compile(_ADAPTED_HELPER,'strict_adapted_epoch_serving','exec'),ns);accepted=ns['validate'](snapshot,_ADAPTED_TRAINING,_ADAPTED_PUBLICATION,_ADAPTED_SHA);del snapshot;gc.collect();return accepted

@torch.no_grad()
def _adapted_qualify():
    import tarfile
    assert not Path('/kaggle/working/submission.csv').exists()
    code={'__name__':'actual_original_pixel_codec'};exec(compile(_ADAPTED_CODEC,'actual_original_pixel_codec','exec'),code);mix={'__name__':'actual_checkpointed_whole94_encoder'};exec(compile(_ADAPTED_MIXIN,'actual_checkpointed_whole94_encoder','exec'),mix)
    class TrainingGraph(mix['CheckpointedEncode'],RaptorClassifier):pass
    p=_adapted_locate('index.private.json',21324,'3a1567989bf65779aeb19785dafdedef7386fb6bedba5ec45da6d3a67a9a79e3');rows=json.loads(p.read_text())['records'];assert len(rows)==32
    devices=[]
    with tarfile.open(p.parent/'pixels.tar','r:') as tar:
        members={m.name:m for m in tar.getmembers()};assert len(members)==32
        for i in range(2):
            device='cuda:'+str(i);model,res=load_model(str(_adapted_model_path),ARMS[0]['arch'],384,device);reference=TrainingGraph(build_backbone(ARMS[0]['arch'],pretrained=False),F_dim=1024).to(device);reference.load_state_dict(model.state_dict(),strict=True);reference.eval();assert model.state_dict().keys()==reference.state_dict().keys() and all(torch.equal(t,reference.state_dict()[n]) for n,t in model.state_dict().items());maximum=0.
            for row in rows:
                member=members[row['file']];assert member.isreg() and member.size==row['bytes'];handle=tar.extractfile(member);assert handle is not None;volume,mask=code['decode'](handle.read(),row['sha256']);assert hashlib.sha256(volume.tobytes()).hexdigest()==row['volume_sha256'] and hashlib.sha256(mask.tobytes()).hexdigest()==row['mask_sha256'];windows=[];values=[]
                for mirror in (False,True):
                    cpu=eval_windows(volume,mask,94,320,norm=NORM,mirror=mirror);assert hashlib.sha256(cpu.contiguous().numpy().tobytes()).hexdigest()==row['original_mirrored_input_sha256' if mirror else 'original_plain_input_sha256'];gpu=gpu_windows(volume,mask,94,320,device,norm=NORM,mirror=mirror);assert torch.equal(cpu,gpu.cpu());windows.append(gpu)
                    with torch.autocast('cuda',dtype=torch.float16):logits=reference(gpu[None]).float()
                    assert logits.shape==(1,12) and torch.isfinite(logits).all();values.append(logits.sigmoid()[0].cpu().numpy())
                actual=infer_pair_probs(model,*windows,device);expected=.5*(values[0]+values[1]);error=float(np.max(np.abs(actual-expected)));maximum=max(maximum,error);assert error<=1e-6,'Served original class differs from complete trained-model graph';del windows
            devices.append({'index':i,'GPU':torch.cuda.get_device_name(i),'MRI_studies':32,'both_original94_input_views_bit_equal':True,'maximum_original_class_vs_training_graph_probability_error':maximum,'peak_allocated_bytes':torch.cuda.max_memory_allocated(i),'complete_learned_graph_tensor_values_equal':True});del model,reference;torch.cuda.empty_cache()
    assert not Path('/kaggle/working/submission.csv').exists();return {'status':'actual_full_parent_epoch_bothT4_original32_MRI_and_training_graph_qualified','epoch':_ADAPTED_TRAINING['segment'],'model_sha256':_ADAPTED_SHA,'training_source_sha256':_ADAPTED_TRAINING['source_sha256'],'devices':devices,'qualified_studies':32,'qualification_cannot_submit':True,'competition_entry':False,'clinical_accuracy_claim':False}

def _adapted_competition():
    _adapted_qualifying[0]=False;start=time.monotonic();main()
    import pandas as pd
    root=find_test_root();ids=pd.read_csv(str(root)+'/test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist();path=Path('/kaggle/working/submission.csv');out=pd.read_csv(path,dtype={'StudyInstanceUID':str});assert out.StudyInstanceUID.tolist()==ids and len(ids)==len(set(ids)) and list(out.columns)==['StudyInstanceUID',*LAB];assert np.isfinite(out[LAB].values).all() and ((out[LAB].values>=0)&(out[LAB].values<=1)).all()
    if len(ids)==3:assert _adapted_counts=={'started':6,'completed':6,'failed':0,'decode_errors':0,'empty_views':0},'Public model/image fallbacks do not qualify'
    return {'status':'actual_full_parent_epoch_MRI_test_roster_CSV_complete','epoch':_ADAPTED_TRAINING['segment'],'model_sha256':_ADAPTED_SHA,'training_source_sha256':_ADAPTED_TRAINING['source_sha256'],'studies':len(ids),'columns':13,'CSV_bytes':path.stat().st_size,'CSV_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'test_roster_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'prediction_counts':_adapted_counts,'devices':[torch.cuda.get_device_name(i) for i in range(2)],'elapsed_seconds':time.monotonic()-start,'independent_internal_validation':False,'competition_entry':False,'clinical_accuracy_claim':False}

if __name__=='__main__':
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';_adapted_boot();result=_adapted_qualify() if _ADAPTED_MODE=='qualification' else _adapted_competition();path=Path('/kaggle/working')/('licensed-parent-epoch-qualification-accepted.json' if _ADAPTED_MODE=='qualification' else 'licensed-parent-epoch-public-accepted.json');path.write_text(json.dumps(result,sort_keys=True,indent=2));print(json.dumps(result),flush=True)
"""
    for name, value in {
        "MODE_VALUE": mode,
        "TRAINING_VALUE": training,
        "PUBLICATION_VALUE": publication,
        "MODEL_SHA_VALUE": model_sha,
        "SERVING_VALUE": serving,
        "CODEC_VALUE": codec,
        "MIXIN_VALUE": mixin,
    }.items():
        patch = patch.replace(name, repr(value))
    header = (
        "# Fine-tuned licensed MRI parent. Original source prose below is historical.\n"
        "# Actual epoch/source/rights/exposure are bound by appended production receipts.\n"
        "import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\n"
        "from pathlib import Path\nimport torch,hashlib\n"
    )
    text = header + source + "\n" + bounded + "\n" + patch
    compile(text, "qualified_full_parent_epoch_inference", "exec")
    return text, {
        "source_sha256": digest(text),
        "mode": mode,
        "epoch": epoch,
        "model_sha256": model_sha,
        "training_kernel": training["kernel_id"],
        "training_source_sha256": training["source_sha256"],
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "helper_sha256": digest(serving),
        "qualification_cannot_submit": mode == "qualification",
        "new_weight_Dataset_required": False,
        "independent_internal_validation": False,
    }
