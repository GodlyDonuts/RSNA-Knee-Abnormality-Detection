"""Compile a strict context component into the pinned licensed MRI parent.

External source and private reference arrays are supplied on the owning node,
never vendored here. Qualification cannot create a competition CSV.
"""

from __future__ import annotations

import ast
import hashlib

from scripts.kaggle.series_boundary_notebook import GUARD, PARENT_SOURCE_SHA256, WEIGHTS_SHA256

SERVING_SOURCE_SHA256 = "4dbeb46a3e647a840047e6cede75328a9945a57ec2db4261250d03e28452cc34"
BOUNDED_SOURCE_SHA256 = "5576dff1348ebb7fa322c0ce78f7fdf8aa4b9ef1f2a6be2b757765fcc733cd42"


def segment(source, node):
    # Include decorators; get_source_segment alone omits @dataclass/@no_grad.
    first = min([node.lineno, *[d.lineno for d in getattr(node, "decorator_list", [])]])
    return "\n".join(source.splitlines()[first - 1 : node.end_lineno])


def bodies(source, *, selected=None):
    result = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and (
            node.module == "__future__" or (node.module or "").startswith("rsna")
        ):
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        names = [getattr(node, "name", None)]
        if isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if selected is None or any(n in selected for n in names):
            result.append(segment(source, node))
    return "\n\n".join(result)


def build(parent, context, trainer, infer, bounded, component_sha, *, mode, reference_sha=None):
    if hashlib.sha256(parent.encode()).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Exact licensed scored parent required")
    return _build(
        parent,
        context,
        trainer,
        infer,
        bounded,
        component_sha,
        mode=mode,
        reference_sha=reference_sha,
    )


def _build(parent, context, trainer, infer, bounded, component_sha, *, mode, reference_sha=None):
    if (
        mode not in ("qualification", "competition")
        or hashlib.sha256(infer.encode()).hexdigest() != SERVING_SOURCE_SHA256
        or hashlib.sha256(bounded.encode()).hexdigest() != BOUNDED_SOURCE_SHA256
        or not _sha(component_sha)
        or (mode == "qualification" and not _sha(reference_sha))
        or (mode == "competition" and reference_sha is not None)
        or parent.count(GUARD) != 1
        or parent[parent.index(GUARD) + len(GUARD) :].strip()
        or hashlib.sha256(context.encode()).hexdigest()
        != "4ed7cd1978a4acac6c233c734914dbe713aed818f229eabd7f23fa10580fbee9"
        or hashlib.sha256(trainer.encode()).hexdigest()
        != "5addea8212501ba31021c739543a7fa627cf70066ad439fa0889ba5c5b3fa59f"
    ):
        raise ValueError(
            "Known mode, immutable head engine and exact final component identity required"
        )
    source = parent[: parent.index(GUARD)]
    replacements = {
        "nproc = max(4, min(96, (os.cpu_count() or 8) - 8))": "nproc = 2",
        "pool.imap_unordered(_pre_one, test_ids, chunksize=2)": "bounded_results(pool, _pre_one, test_ids, limit=4)",
        "q = queue.Queue(maxsize=QUEUE_MAX)": "q = queue.Queue(maxsize=2)",
        'print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)': 'print(json.dumps({"context_MRI_fallback_index":i,"error_type":type(e).__name__}),flush=True)',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("Unique original bounded queue and sanitized fallback sites required")
        source = source.replace(old, new)
    serving_names = {
        "FORMAT",
        "SOURCE_GIT",
        "PARENT_SOURCE",
        "HELPER_SHA",
        "POPULATION_SHA",
        "LABEL_SOURCE_SHA",
        "FEATURE_SOURCE_SHA",
        "PROVENANCE_KEYS",
        "sha256",
        "is_sha",
        "model_from_component",
        "load_component",
        "predict_context_pair",
    }
    helper = (
        "from pathlib import Path\nfrom dataclasses import dataclass,asdict\n"
        + "PARENT_SHA = "
        + repr(WEIGHTS_SHA256)
        + "\nTARGETS = tuple(LAB)\n"
        + bodies(context)
        + "\n"
        + bodies(trainer, selected={"Recipe"})
        + "\n"
        + bodies(infer, selected=serving_names)
        + "\n"
        + bodies(bounded)
    )
    patch = r"""
_CONTEXT_MODE = MODE_VALUE
_CONTEXT_COMPONENT_SHA = COMPONENT_VALUE
_CONTEXT_REFERENCE_SHA = REFERENCE_VALUE
_CONTEXT_PARENTSOURCE_SHA = PARENT_VALUE
_context_models = {}
_context_lock = threading.Lock()
_context_counts = {'pairs_started':0,'pairs_completed':0,'head_fallbacks':0,'empty_studies':0}
_context_max_correction = [0.0]
_context_original_gpu_windows = gpu_windows

# The legacy decoder's fallback semantics remain explicit, but no case identifier
# or exception text is added to stdout by this adaptation.
def _pre_one(sid):
    try:
        vol,mask = _READER(sid,_SER,_TSDIR)
    except Exception as exc:
        print(json.dumps({'context_preprocessing_error_type':type(exc).__name__}),flush=True)
        vol=np.zeros((MAXS,IMG,IMG),np.uint8);mask=np.zeros(MAXS,np.uint8)
    return sid,vol,mask

def gpu_windows(vol,mask,k,res,dev,norm=NORM,mirror=False):
    assert k==94 and res==320 and vol.shape==(96,384,384)
    values=_context_original_gpu_windows(vol,mask,k,res,dev,norm=norm,mirror=mirror)
    cs=np.asarray([max(1,min(c,94)) for c in _eval_centers(mask,96,94)],dtype=np.int64)
    return {'windows':values,'centers':torch.from_numpy(cs),'present':torch.from_numpy(mask[cs].astype(bool)),'mirror':bool(mirror)}

@torch.no_grad()
def infer_pair_probs(model,plain,mirrored,device):
    assert plain['mirror'] is False and mirrored['mirror'] is True
    assert torch.equal(plain['centers'],mirrored['centers']) and torch.equal(plain['present'],mirrored['present'])
    assert all(not m.training for m in model.modules())
    with _context_lock:_context_counts['pairs_started']+=1
    features,parents=[],[]
    with torch.cuda.device(device):
        for bundle in (plain,mirrored):
            with torch.autocast('cuda',dtype=torch.float16):
                f=model.encode(bundle['windows'][None]);p=model.head(f).float()
            assert f.shape==(1,94,1024) and f.dtype==torch.float32 and p.shape==(1,12) and torch.isfinite(f).all() and torch.isfinite(p).all()
            features.append(f[0]);parents.append(p[0])
        parent_logits=torch.stack(parents)
        original_probability=parent_logits.sigmoid().mean(0)
        if not plain['present'].any():
            result=original_probability
            with _context_lock:_context_counts['empty_studies']+=1
        else:
            try:
                probability,correction=predict_context_pair(_context_models[str(device)],torch.stack(features),parent_logits,plain['centers'],plain['present'])
                assert probability.shape==(12,) and torch.isfinite(probability).all() and (correction.abs()<=0.5).all()
                result=probability
                with _context_lock:_context_max_correction[0]=max(_context_max_correction[0],float(correction.abs().max()))
            except Exception as exc:
                # Reuse already-computed parent logits; never a second CNN pass.
                result=original_probability
                with _context_lock:_context_counts['head_fallbacks']+=1
                print(json.dumps({'context_head_fallback_error_type':type(exc).__name__}),flush=True)
    with _context_lock:_context_counts['pairs_completed']+=1
    return result.cpu().numpy()

def _context_boot():
    assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert len(ARMS)==1 and K_EVAL==94 and INPUT_RES==320 and SLOT_SIZES==[26,22,18,12,18] and tuple(LAB)==tuple(TARGETS)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    path=find_weight_file('context-head-e10-'+_CONTEXT_COMPONENT_SHA[:12]+'.pt')
    for i in range(2):
        dev='cuda:'+str(i);_context_models[dev],p=load_component(path,_CONTEXT_COMPONENT_SHA,device=dev)
        assert p['parent_weights_sha256']==PARENT_SHA and p['parent_source_sha256']==_CONTEXT_PARENTSOURCE_SHA
    return path

def _context_qualify():
    # Model-head-only precision: cached real MRI features from the already
    # complete32-study source, not another CNN decode or diagnostic model.
    assert not Path('/kaggle/working/submission.csv').exists()
    index_sha='a1bba9e618444f16ddd453efa1bb8650245f8259dcd9a00348b17efd87c1e100'
    candidates=[]
    for d in Path('/kaggle/input').iterdir():
        if not d.is_dir() or 'competition' in d.name or 'abnormality-detection' in d.name:continue
        if any((d/x).exists() for x in ('train.csv','test.csv','train_images','train_series')):continue
        for root,dirs,files in os.walk(d):
            if len(Path(root).relative_to(d).parts)>=5:dirs[:]=[]
            if 'index.private.json' in files:
                p=Path(root)/'index.private.json'
                if p.stat().st_size==8574 and sha256(p)==index_sha:candidates.append(p)
    assert candidates
    index_path=candidates[0];index=json.loads(index_path.read_text());rows=index['records'];assert len(rows)==32 and index['weights_sha256']==PARENT_SHA
    reference_path=find_weight_file('context-cpu-reference-'+_CONTEXT_REFERENCE_SHA[:12]+'.npy');assert sha256(reference_path)==_CONTEXT_REFERENCE_SHA
    reference=np.load(reference_path,allow_pickle=False);assert reference.shape==(32,12) and reference.dtype==np.float32 and np.isfinite(reference).all()
    devices=[]
    for i in range(2):
        dev='cuda:'+str(i);m=_context_models[dev];errors=[];largest=0.0;started=time.monotonic()
        with torch.cuda.device(dev):
            torch.cuda.reset_peak_memory_stats(dev)
            for j,row in enumerate(rows):
                p=index_path.parent/row['file'];assert p.stat().st_size==row['bytes'] and sha256(p)==row['sha256']
                with np.load(p,allow_pickle=False) as a:
                    probability,correction=predict_context_pair(m,torch.from_numpy(a['features']).to(dev),torch.from_numpy(a['parent_logits']).to(dev),torch.from_numpy(a['centers']),torch.from_numpy(a['present']))
                errors.append(float(np.abs(probability.cpu().numpy()-reference[j]).max()));largest=max(largest,float(correction.abs().max()))
            torch.cuda.synchronize(dev)
            assert max(errors)<=1e-5 and largest>1e-8
            devices.append({'index':i,'name':torch.cuda.get_device_name(i),'original_MRI_feature_studies':32,'max_CPU_GPU_probability_error':max(errors),'max_abs_nonzero_correction':largest,'elapsed_seconds':time.monotonic()-started,'peak_GPU_bytes':torch.cuda.max_memory_allocated(dev)})
    result={'status':'fixed_final_context_head_actual_twoT4_precision_accepted','component_sha256':_CONTEXT_COMPONENT_SHA,'parent_checkpoint_sha256':PARENT_SHA,'original_feature_index_sha256':index_sha,'CPU_reference_sha256':_CONTEXT_REFERENCE_SHA,'devices':devices,'completed_training_updates':2430,'fitted_studies':3887,'clinical_or_competition_accuracy':False,'new_CNN_study_encodings':0,'competition_submission':False}
    Path('/kaggle/working/context-head-precision-accepted.json').write_text(json.dumps(result,sort_keys=True,indent=2));assert not Path('/kaggle/working/submission.csv').exists();print(json.dumps(result),flush=True)

def _context_competition():
    started=time.monotonic()
    main()
    import pandas as pd
    root=find_test_root();ids=pd.read_csv(root+'/test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist();path=Path('/kaggle/working/submission.csv');value=pd.read_csv(path,dtype={'StudyInstanceUID':str})
    assert value.StudyInstanceUID.tolist()==ids and len(set(ids))==len(ids) and list(value.columns)==['StudyInstanceUID',*LAB]
    assert np.isfinite(value[LAB].values).all() and ((value[LAB].values>=0)&(value[LAB].values<=1)).all()
    if len(ids)==3:assert _context_counts=={'pairs_started':3,'pairs_completed':3,'head_fallbacks':0,'empty_studies':0},'A public fallback is not successful MRI acceptance'
    result={'status':'fixed_final_context_MRI_CSV_complete','parent_source_sha256':_CONTEXT_PARENTSOURCE_SHA,'parent_checkpoint_sha256':PARENT_SHA,'context_component_sha256':_CONTEXT_COMPONENT_SHA,'completed_training_updates':2430,'fitted_studies':3887,'studies':len(ids),'columns':13,'CSV_bytes':path.stat().st_size,'CSV_sha256':sha256(path),'test_roster_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'counts':_context_counts,'max_abs_context_correction':_context_max_correction[0],'encoder_precision':'original_FP16','context_precision':'FP32_no_TF32','pending_CPU_tasks_cap':4,'preprocessing_CPUs':2,'elapsed_seconds':time.monotonic()-started,'attributed_external_parent':True,'independent_internal_validation':False,'clinical_accuracy_claim':False,'competition_submission':False}
    Path('/kaggle/working/context-MRI-accepted.json').write_text(json.dumps(result,sort_keys=True,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    _context_boot()
    if _CONTEXT_MODE=='qualification':_context_qualify()
    elif _CONTEXT_MODE=='competition':_context_competition()
    else:raise ValueError('Unknown fixed context operation')
"""
    for name, value in {
        "MODE_VALUE": mode,
        "COMPONENT_VALUE": component_sha,
        "REFERENCE_VALUE": reference_sha,
        "PARENT_VALUE": PARENT_SOURCE_SHA256,
    }.items():
        patch = patch.replace(name, repr(value))
    text = source + "\n" + helper + "\n" + patch
    compile(text, "licensed_fixed_final_context_notebook", "exec")
    return text, {
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_bytes": len(text.encode()),
        "mode": mode,
        "component_sha256": component_sha,
        "parent_checkpoint_sha256": WEIGHTS_SHA256,
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "helper_sha256": hashlib.sha256(context.encode()).hexdigest(),
        "serving_module_sha256": hashlib.sha256(infer.encode()).hexdigest(),
        "trainer_source_sha256": hashlib.sha256(trainer.encode()).hexdigest(),
        "reference_sha256": reference_sha,
        "qualification_cannot_submit": mode == "qualification",
        "independent_internal_validation": False,
    }


def _sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )
