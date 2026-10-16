"""Compile either matched spatial role into the exact licensed scored parent.

The supplied external source stays private. Qualification reads the already
accepted32-study feature archive; it cannot call MRI inference or create a CSV.
"""

from __future__ import annotations

import hashlib

from scripts.kaggle.released_context_notebook import bodies
from scripts.kaggle.series_boundary_notebook import GUARD, PARENT_SOURCE_SHA256, WEIGHTS_SHA256

SOURCE_HASHES = {
    "context": "4ed7cd1978a4acac6c233c734914dbe713aed818f229eabd7f23fa10580fbee9",
    "trainer": "5addea8212501ba31021c739543a7fa627cf70066ad439fa0889ba5c5b3fa59f",
    "summaries": "d2c45353e56b8c7a86a0b83a15edb35395341867d72ec68b91d77e59b705be17",
    "regional": "1b8346a4b9772d3dede817089b8f26204d17854f035556e4ba9ef8c5944cf4ea",
    "infer": "141886c0cd4e5836d45d2d6b5d491076fc73b106c58a9e47a3a13ce64f4c7e25",
    "common": "4dbeb46a3e647a840047e6cede75328a9945a57ec2db4261250d03e28452cc34",
    "bounded": "5576dff1348ebb7fa322c0ce78f7fdf8aa4b9ef1f2a6be2b757765fcc733cd42",
}


def _sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def build(parent, sources, component_sha, *, role, mode, reference_sha=None):
    if hashlib.sha256(parent.encode()).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Exact licensed scored parent source required")
    return _build(parent, sources, component_sha, role=role, mode=mode, reference_sha=reference_sha)


def _build(parent, sources, component_sha, *, role, mode, reference_sha=None):
    if (
        set(sources) != set(SOURCE_HASHES)
        or any(
            hashlib.sha256(sources[k].encode()).hexdigest() != h for k, h in SOURCE_HASHES.items()
        )
        or role not in ("global_repeat", "regional")
        or mode not in ("qualification", "competition")
        or not _sha(component_sha)
        or (mode == "qualification" and not _sha(reference_sha))
        or (mode == "competition" and reference_sha is not None)
        or parent.count(GUARD) != 1
        or parent[parent.index(GUARD) + len(GUARD) :].strip()
    ):
        raise ValueError(
            "Exact matched role, immutable sources, final bytes and known mode required"
        )
    source = parent[: parent.index(GUARD)]
    replacements = {
        "nproc = max(4, min(96, (os.cpu_count() or 8) - 8))": "nproc = 2",
        "pool.imap_unordered(_pre_one, test_ids, chunksize=2)": "bounded_results(pool, _pre_one, test_ids, limit=4)",
        "q = queue.Queue(maxsize=QUEUE_MAX)": "q = queue.Queue(maxsize=2)",
        'print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)': 'print(json.dumps({"spatial_MRI_fallback_index":i,"error_type":type(e).__name__}),flush=True)',
    }
    for before, after in replacements.items():
        if source.count(before) != 1:
            raise ValueError("Original unique bounded queue and sanitized fallback sites required")
        source = source.replace(before, after)
    helper = (
        "from pathlib import Path\nfrom dataclasses import dataclass,asdict\n"
        + "PARENT_SHA="
        + repr(WEIGHTS_SHA256)
        + "\nTARGETS=tuple(LAB)\n"
        + bodies(sources["context"])
        + "\n"
        + bodies(sources["trainer"], selected={"Recipe"})
        + "\n"
        + bodies(sources["summaries"])
        + "\n"
        + bodies(
            sources["regional"],
            selected={
                "ENGINE",
                "GENERIC_TRAINER_SHA",
                "PARENT_SOURCE",
                "POPULATIONS",
                "REGIONAL_HELPER_SHA",
                "REPRESENTATION",
                "ROLES",
                "feature_vector",
            },
        )
        + "\n"
        + bodies(sources["common"], selected={"sha256", "is_sha"})
        + "\n"
        + bodies(
            sources["infer"],
            selected={
                "FORMAT",
                "SOURCE_GIT",
                "RECOVERY_SHA",
                "FIXED_PROVENANCE",
                "model_from_component",
                "load_component",
                "predict_pair",
            },
        )
        + "\n"
        + bodies(sources["bounded"])
    )
    patch = r"""
_SPATIAL_MODE=MODE_VALUE
_SPATIAL_ROLE=ROLE_VALUE
_SPATIAL_COMPONENT_SHA=COMPONENT_VALUE
_SPATIAL_REFERENCE_SHA=REFERENCE_VALUE
_SPATIAL_PARENT_SOURCE=PARENT_VALUE
_spatial_models={};_spatial_captures={};_spatial_verified_forwards=set();_spatial_pixels=set()
_spatial_lock=threading.Lock()
_spatial_counts={'pairs_started':0,'pairs_completed':0,'head_fallbacks':0,'empty_studies':0}
_spatial_max_correction=[0.0]
_spatial_original_windows=gpu_windows

def _pre_one(sid):
    try:vol,mask=_READER(sid,_SER,_TSDIR)
    except Exception as exc:
        print(json.dumps({'spatial_preprocessing_error_type':type(exc).__name__}),flush=True)
        vol=np.zeros((MAXS,IMG,IMG),np.uint8);mask=np.zeros(MAXS,np.uint8)
    return sid,vol,mask

def gpu_windows(vol,mask,k,res,dev,norm=NORM,mirror=False):
    assert k==94 and res==320 and vol.shape==(96,384,384)
    windows=_spatial_original_windows(vol,mask,k,res,dev,norm=norm,mirror=mirror)
    key=(str(dev),bool(mirror))
    if key not in _spatial_pixels:
        assert torch.equal(windows.cpu(),eval_windows(vol,mask,k,res,norm=norm,mirror=mirror))
        with _spatial_lock:_spatial_pixels.add(key)
    cs=np.asarray([max(1,min(c,94)) for c in _eval_centers(mask,96,94)],dtype=np.int64)
    return {'windows':windows,'centers':torch.from_numpy(cs),'present':torch.from_numpy(mask[cs].astype(bool)),'mirror':bool(mirror)}

@torch.no_grad()
def infer_pair_probs(model,plain,mirrored,device):
    assert plain['mirror'] is False and mirrored['mirror'] is True
    assert torch.equal(plain['centers'],mirrored['centers']) and torch.equal(plain['present'],mirrored['present'])
    assert all(not m.training for m in model.modules())
    key=(str(device),id(model))
    with _spatial_lock:
        _spatial_counts['pairs_started']+=1
        if key not in _spatial_captures:_spatial_captures[key]=RegionalCapture(model.backbone)
        capture=_spatial_captures[key]
    features=[];regions=[];parents=[]
    with torch.cuda.device(device):
        try:
            for bundle in (plain,mirrored):
                capture.begin()
                with torch.autocast('cuda',dtype=torch.float16):
                    f=model.encode(bundle['windows'][None]);p=model.head(f).float()
                regional=capture.end(94)
                assert f.shape==(1,94,1024) and f.dtype==torch.float32 and p.shape==(1,12) and torch.isfinite(f).all() and torch.isfinite(p).all()
                if key not in _spatial_verified_forwards:
                    with torch.autocast('cuda',dtype=torch.float16):
                        unhooked=model.encode(bundle['windows'][None]);unhooked_p=model.head(unhooked).float()
                    assert torch.equal(f,unhooked) and torch.equal(p,unhooked_p),'Observer changed the original parent forward'
                features.append(f[0]);regions.append(regional);parents.append(p[0])
            with _spatial_lock:_spatial_verified_forwards.add(key)
        except Exception:
            # Do not leave an active hook/cache to poison a later original study.
            capture.handle.remove()
            with _spatial_lock:_spatial_captures.pop(key,None)
            raise
        parent_logits=torch.stack(parents);original_probability=parent_logits.sigmoid().mean(0)
        if not plain['present'].any():
            result=original_probability
            with _spatial_lock:_spatial_counts['empty_studies']+=1
        else:
            try:
                probability,correction=predict_pair(_spatial_models[str(device)],_SPATIAL_ROLE,torch.stack(features),torch.stack(regions),parent_logits,plain['centers'].to(device),plain['present'].to(device))
                assert probability.shape==(12,) and torch.isfinite(probability).all() and (correction.abs()<=.5).all()
                result=probability
                with _spatial_lock:_spatial_max_correction[0]=max(_spatial_max_correction[0],float(correction.abs().max()))
            except Exception as exc:
                result=original_probability
                with _spatial_lock:_spatial_counts['head_fallbacks']+=1
                print(json.dumps({'spatial_head_fallback_error_type':type(exc).__name__}),flush=True)
    with _spatial_lock:_spatial_counts['pairs_completed']+=1
    return result.cpu().numpy()

def _spatial_boot():
    assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert len(ARMS)==1 and K_EVAL==94 and INPUT_RES==320 and SLOT_SIZES==[26,22,18,12,18] and tuple(LAB)==tuple(TARGETS)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    path=find_weight_file('spatial-'+_SPATIAL_ROLE+'-e10-'+_SPATIAL_COMPONENT_SHA[:12]+'.pt')
    for i in range(2):
        dev='cuda:'+str(i);_spatial_models[dev],p=load_component(path,_SPATIAL_COMPONENT_SHA,_SPATIAL_ROLE,device=dev)
        assert p['parent_weights_sha256']==PARENT_SHA and p['parent_source_sha256']==_SPATIAL_PARENT_SOURCE

def _spatial_qualify():
    import io,tarfile
    assert not Path('/kaggle/working/submission.csv').exists()
    index_sha='280b3d04e4506a1fc0771280a16a6c1a9fe6ae3455ffdac8fbbc048f2a343481';candidates=[]
    for d in Path('/kaggle/input').iterdir():
        if not d.is_dir() or 'competition' in d.name or 'abnormality-detection' in d.name:continue
        if any((d/x).exists() for x in ('train.csv','test.csv','train_images','train_series')):continue
        for root,dirs,files in os.walk(d):
            if len(Path(root).relative_to(d).parts)>=5:dirs[:]=[]
            if 'index.private.json' in files:
                p=Path(root)/'index.private.json'
                if p.stat().st_size==9787 and sha256(p)==index_sha:candidates.append(p)
    assert candidates;index_path=candidates[0];index=json.loads(index_path.read_text());rows=index['records'];archive=index_path.parent/'features.tar'
    assert len(rows)==32 and index['representation']==REPRESENTATION and index['source_sha256']==REGIONAL_HELPER_SHA and index['parent_source_sha256']==PARENT_SOURCE
    assert archive.stat().st_size==111052800 and sha256(archive)=='cc2f06ec819a707bc72c55c0a16778493d428a450130b3f8abb0bc6eee0aedee'
    reference_path=find_weight_file('spatial-'+_SPATIAL_ROLE+'-cpu-reference-'+_SPATIAL_REFERENCE_SHA[:12]+'.npy');assert sha256(reference_path)==_SPATIAL_REFERENCE_SHA
    reference=np.load(reference_path,allow_pickle=False);assert reference.shape==(32,12) and reference.dtype==np.float32 and np.isfinite(reference).all()
    devices=[]
    for i in range(2):
        dev='cuda:'+str(i);errors=[];largest=0.;started=time.monotonic()
        with torch.cuda.device(dev),tarfile.open(archive,'r:') as tar:
            torch.cuda.reset_peak_memory_stats(dev);members={m.name:m for m in tar.getmembers()};assert len(members)==32 and all(m.isfile() for m in members.values())
            for j,row in enumerate(rows):
                member=members[row['file']];assert member.size==row['bytes'] and member.size<4*2**20;raw=tar.extractfile(member).read();assert hashlib.sha256(raw).hexdigest()==row['sha256']
                with np.load(io.BytesIO(raw),allow_pickle=False) as a:
                    probability,correction=predict_pair(_spatial_models[dev],_SPATIAL_ROLE,*[torch.from_numpy(a[k]).to(dev) for k in ('features','regional','parent_logits','centers','present')])
                errors.append(float(np.abs(probability.cpu().numpy()-reference[j]).max()));largest=max(largest,float(correction.abs().max()))
            torch.cuda.synchronize(dev);assert max(errors)<=1e-5 and largest>1e-8
            devices.append({'index':i,'name':torch.cuda.get_device_name(i),'original_MRI_feature_studies':32,'max_CPU_GPU_probability_error':max(errors),'max_abs_nonzero_correction':largest,'elapsed_seconds':time.monotonic()-started,'peak_GPU_bytes':torch.cuda.max_memory_allocated(dev)})
    result={'status':'fixed_final_spatial_head_actual_twoT4_precision_accepted','role':_SPATIAL_ROLE,'component_sha256':_SPATIAL_COMPONENT_SHA,'parent_checkpoint_sha256':PARENT_SHA,'original_feature_index_sha256':index_sha,'CPU_reference_sha256':_SPATIAL_REFERENCE_SHA,'devices':devices,'completed_training_updates':2430,'fitted_studies':3887,'clinical_or_competition_accuracy':False,'new_CNN_study_encodings':0,'competition_submission':False}
    Path('/kaggle/working/spatial-head-precision-accepted.json').write_text(json.dumps(result,sort_keys=True,indent=2));assert not Path('/kaggle/working/submission.csv').exists();print(json.dumps(result),flush=True)

def _spatial_competition():
    started=time.monotonic();main()
    import pandas as pd
    root=find_test_root();ids=pd.read_csv(root+'/test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist();path=Path('/kaggle/working/submission.csv');value=pd.read_csv(path,dtype={'StudyInstanceUID':str})
    assert value.StudyInstanceUID.tolist()==ids and len(set(ids))==len(ids) and list(value.columns)==['StudyInstanceUID',*LAB]
    assert np.isfinite(value[LAB].values).all() and ((value[LAB].values>=0)&(value[LAB].values<=1)).all()
    if len(ids)==3:assert _spatial_counts=={'pairs_started':3,'pairs_completed':3,'head_fallbacks':0,'empty_studies':0},'Public fallback is not successful MRI acceptance'
    result={'status':'fixed_final_spatial_MRI_CSV_complete','role':_SPATIAL_ROLE,'parent_source_sha256':_SPATIAL_PARENT_SOURCE,'parent_checkpoint_sha256':PARENT_SHA,'component_sha256':_SPATIAL_COMPONENT_SHA,'completed_training_updates':2430,'fitted_studies':3887,'studies':len(ids),'columns':13,'CSV_bytes':path.stat().st_size,'CSV_sha256':sha256(path),'test_roster_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'counts':_spatial_counts,'max_abs_context_correction':_spatial_max_correction[0],'encoder_precision':'original_FP16','context_precision':'FP32_no_TF32','regional_storage_precision':'float16','pixel_checks':len(_spatial_pixels),'parent_observer_exact_devices':len(_spatial_verified_forwards),'pending_CPU_tasks_cap':4,'preprocessing_CPUs':2,'elapsed_seconds':time.monotonic()-started,'attributed_external_parent':True,'independent_internal_validation':False,'clinical_accuracy_claim':False,'competition_submission':False}
    Path('/kaggle/working/spatial-MRI-accepted.json').write_text(json.dumps(result,sort_keys=True,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':
    _spatial_boot()
    if _SPATIAL_MODE=='qualification':_spatial_qualify()
    elif _SPATIAL_MODE=='competition':_spatial_competition()
    else:raise ValueError('Unknown fixed spatial operation')
"""
    for key, value in {
        "MODE_VALUE": mode,
        "ROLE_VALUE": role,
        "COMPONENT_VALUE": component_sha,
        "REFERENCE_VALUE": reference_sha,
        "PARENT_VALUE": PARENT_SOURCE_SHA256,
    }.items():
        patch = patch.replace(key, repr(value))
    result = source + "\n" + helper + "\n" + patch
    compile(result, "licensed_fixed_final_spatial_notebook", "exec")
    return result, {
        "source_sha256": hashlib.sha256(result.encode()).hexdigest(),
        "source_bytes": len(result.encode()),
        "mode": mode,
        "role": role,
        "component_sha256": component_sha,
        "reference_sha256": reference_sha,
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "parent_checkpoint_sha256": WEIGHTS_SHA256,
        "source_module_sha256": dict(SOURCE_HASHES),
        "qualification_cannot_submit": mode == "qualification",
        "independent_internal_validation": False,
    }
