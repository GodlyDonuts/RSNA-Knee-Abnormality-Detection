"""Observe spatial summaries without changing the pinned scored parent forward.

Regional and global features/logits are cached from the SAME original forward.
The earlier cache is an identity/metadata and probability reference, not an
interchangeable global embedding. No label/optimizer/CSV.
"""

from __future__ import annotations

import hashlib
import json

from scripts.kaggle.released_context_notebook import bodies
from scripts.kaggle.series_boundary_notebook import GUARD, PARENT_SOURCE_SHA256

CONTEXT_HELPER_SHA = "4ed7cd1978a4acac6c233c734914dbe713aed818f229eabd7f23fa10580fbee9"
BOUNDED_HELPER_SHA = "5576dff1348ebb7fa322c0ce78f7fdf8aa4b9ef1f2a6be2b757765fcc733cd42"


def build(parent, context, regional, bounded, roster):
    if hashlib.sha256(parent.encode()).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Exact previously scored licensed parent source required")
    return _build(parent, context, regional, bounded, roster)


def _build(parent, context, regional, bounded, roster):
    if (
        len(roster) != 32
        or len(set(roster)) != 32
        or roster != sorted(roster)
        or not all(isinstance(u, str) and u for u in roster)
        or parent.count(GUARD) != 1
        or parent[parent.index(GUARD) + len(GUARD) :].strip()
        or hashlib.sha256(context.encode()).hexdigest() != CONTEXT_HELPER_SHA
        or hashlib.sha256(bounded.encode()).hexdigest() != BOUNDED_HELPER_SHA
    ):
        raise ValueError(
            "Fixed32engineering roster, exact helpers and unique parent guard required"
        )
    patch = r"""
_REGIONAL_ROSTER = ROSTER_VALUE
_REGIONAL_SOURCE = SOURCE_VALUE
_REGIONAL_PARENT = PARENT_VALUE
_REGIONAL_OLD_INDEX = 'a1bba9e618444f16ddd453efa1bb8650245f8259dcd9a00348b17efd87c1e100'

def _regional_old_cache():
    candidates=[]
    for d in Path('/kaggle/input').iterdir():
        if not d.is_dir() or 'competition' in d.name or 'abnormality-detection' in d.name:continue
        if any((d/x).exists() for x in ('train.csv','test.csv','train_images','train_series')):continue
        for root,dirs,files in os.walk(d):
            if len(Path(root).relative_to(d).parts)>=5:dirs[:]=[]
            if 'index.private.json' in files:
                p=Path(root)/'index.private.json'
                if p.stat().st_size==8574 and hashlib.sha256(p.read_bytes()).hexdigest()==_REGIONAL_OLD_INDEX:candidates.append(p)
    assert candidates
    path=candidates[0];value=json.loads(path.read_text());assert value['weights_sha256']=='7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
    assert [r['study_uid'] for r in value['records']]==_REGIONAL_ROSTER
    result={}
    for row in value['records']:
        p=path.parent/row['file'];assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
        with np.load(p,allow_pickle=False) as a:result[row['study_uid']]={n:a[n].copy() for n in ('features','parent_logits','centers','present')}
    return result

def _regional_pre_one(sid):
    try:return sid,*_READER(sid,_SER,_TSDIR)
    except Exception:return sid,None,None

def _regional_probe_main():
    import io,multiprocessing as mp,tarfile
    import pandas as pd
    started=time.monotonic();assert not Path('/kaggle/working/submission.csv').exists()
    assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert K_EVAL==94 and INPUT_RES==320 and SLOT_SIZES==list(SIZES) and len(ARMS)==1
    torch.set_num_threads(1);old=_regional_old_cache()
    out=Path('/kaggle/working/released-regional-features');out.mkdir(mode=0o700)
    archive_path=out/'features.tar';archive=tarfile.open(archive_path,'x:')
    root=Path(find_test_root());series=pd.read_csv(root/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str});series=series.loc[series.StudyInstanceUID.isin(_REGIONAL_ROSTER)];assert set(series.StudyInstanceUID)==set(_REGIONAL_ROSTER)
    global _SER,_TSDIR,_READER
    _SER={k:v.to_dict('records') for k,v in series.groupby('StudyInstanceUID')};_TSDIR=str(root/('train_series' if (root/'train_series').is_dir() else 'train_images'));assert Path(_TSDIR).is_dir()
    devices=['cuda:0','cuda:1'];models={d:load_model(find_weight_file(ARMS[0]['file']),ARMS[0]['arch'],ARMS[0]['res'],d)[0] for d in devices};captures={d:RegionalCapture(models[d].backbone) for d in devices}
    q=queue.Queue(maxsize=2);lock=threading.Lock();records=[];failures=[];qualified={};payload_bytes=[0];feature_error=[0.];prob_error=[0.];old_exact=[0]
    def consumer(device):
        with torch.cuda.device(device),torch.no_grad():
            model=models[device];capture=captures[device];torch.cuda.reset_peak_memory_stats(device)
            while True:
                item=q.get()
                try:
                    if item is None:return
                    stage='decoded_inputs';debug={};sid,vol,mask=item;assert vol is not None and mask is not None and mask.sum()>0
                    cs=np.asarray([max(1,min(c,94)) for c in _eval_centers(mask,96,94)],dtype=np.int64);centers=torch.from_numpy(cs);slots,depth=window_metadata(centers);_,mirrored_depth=window_metadata(centers,mirrored=True)
                    assert np.array_equal(cs,old[sid]['centers']) and np.array_equal(mask[cs].astype(bool),old[sid]['present'])
                    local=[];global_features=[];parent_logits=[]
                    for view,mirror in enumerate((False,True)):
                        stage='original_pixels'
                        x=gpu_windows(vol,mask,94,320,device,norm=NORM,mirror=mirror)
                        if device not in qualified:assert torch.equal(x.cpu(),eval_windows(vol,mask,94,320,norm=NORM,mirror=mirror))
                        stage='regional_encode';capture.begin()
                        with torch.autocast('cuda',dtype=torch.float16):f=model.encode(x[None]);p=model.head(f).float()
                        regions=capture.end(94);assert f.shape==(1,94,1024) and regions.dtype==torch.float16 and torch.isfinite(p).all()
                        if device not in qualified:
                            stage='unhooked_original_forward'
                            with torch.autocast('cuda',dtype=torch.float16):without=model.encode(x[None]);without_p=model.head(without).float()
                            assert torch.equal(f,without) and torch.equal(p,without_p),'Observer changed original CNN/head forward'
                        stage='old_probability_reference';actual=f[0].float().cpu().numpy();reference=old[sid]['features'][view];error=float((np.abs(actual-reference)/np.maximum(1,np.abs(reference))).max());pp=p[0].sigmoid().cpu().numpy();prior=torch.from_numpy(old[sid]['parent_logits'][view]).sigmoid().numpy();pe=float(np.abs(pp-prior).max());debug.update(old_feature_scaled_error=error,old_probability_error=pe);assert pe<=.003
                        with lock:feature_error[0]=max(feature_error[0],error);prob_error[0]=max(prob_error[0],pe);old_exact[0]+=int(np.array_equal(actual,reference))
                        local.append(regions.cpu().numpy());global_features.append(actual);parent_logits.append(p[0].cpu().numpy())
                    stage='coherent_archive_write';arrays={'regional':np.stack(local),'features':np.stack(global_features),'parent_logits':np.stack(parent_logits),'centers':cs,'slots':slots.numpy(),'depth':np.stack([depth.numpy(),mirrored_depth.numpy()]),'present':mask[cs].astype(bool)};assert arrays['regional'].shape==(2,94,7,1024) and arrays['features'].shape==(2,94,1024) and arrays['features'].dtype==np.float32 and arrays['parent_logits'].shape==(2,12)
                    handle=io.BytesIO();np.savez(handle,**arrays);raw=handle.getvalue();assert len(raw)<4*2**20;name=hashlib.sha256(sid.encode()).hexdigest()+'.npz'
                    with lock:
                        payload_bytes[0]+=len(raw);assert payload_bytes[0]<128*2**20
                        info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600;archive.addfile(info,io.BytesIO(raw));records.append({'study_uid':sid,'file':name,'archive':'features.tar','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()});qualified[device]={'device':device,'name':torch.cuda.get_device_name(device),'actual_CPU_GPU_original_pixels_exact':True,'observer_preserves_original_forward_bit_equal':True,'actual_first_chunk_CPU_GPU_regional_scaled_error':capture.maximum_CPU_GPU_scaled_error,'maximum_FP16_storage_scaled_error':capture.maximum_quantization_scaled_error}
                        if len(records)%8==0:print(json.dumps({'regional_feature_studies_complete':len(records),'total':32}),flush=True)
                except Exception as exc:
                    with lock:
                        failure={'stage':stage,'error_type':type(exc).__name__,'scalars':debug};failures.append(failure);print(json.dumps({'regional_probe_failure':failure}),flush=True)
                finally:q.task_done()
    threads=[threading.Thread(target=consumer,args=(d,),daemon=True) for d in devices]
    for t in threads:t.start()
    with mp.get_context('fork').Pool(2,initializer=_pre_init) as pool:
        for item in bounded_results(pool,_regional_pre_one,_REGIONAL_ROSTER,limit=4):
            assert time.monotonic()-started<2700;q.put(item)
    for _ in threads:q.put(None)
    q.join()
    for t in threads:t.join()
    archive.close();Path('/kaggle/working/regional-probe-observation.json').write_text(json.dumps({'completed_records':len(records),'failures':failures,'qualified_devices':list(qualified.values()),'actual_acceptance':not failures and len(records)==32 and len(qualified)==2},sort_keys=True));assert not failures and len(records)==32 and len(qualified)==2
    records.sort(key=lambda r:_REGIONAL_ROSTER.index(r['study_uid']));index=out/'index.private.json';index.write_text(json.dumps({'records':records,'regional_version':REGIONAL_VERSION,'representation':'same_forward_global_regional_v1','names':list(REGIONAL_NAMES),'source_sha256':_REGIONAL_SOURCE,'parent_source_sha256':_REGIONAL_PARENT,'old_global_index_sha256':_REGIONAL_OLD_INDEX,'clinical_labels_reports_read':False},sort_keys=True));os.chmod(index,0o600)
    def sha_file(path):
        with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    receipt={'status':'actual32_original_MRI_regional_feature_probe_accepted','studies':32,'views':2,'windows_per_view':94,'region_summary_count':7,'channels_retained':1024,'compute_dtype':'float32','storage_dtype':'float16','global_feature_dtype':'float32','same_forward_global_regional_and_parent_logits':True,'old_global_embeddings_used_as_new_inputs':False,'old_feature_difference_is_acceptance_criterion':False,'regional_names':list(REGIONAL_NAMES),'devices':list(qualified.values()),'original_global_feature_views_bit_equal_to_existing_cache':old_exact[0],'maximum_original_global_feature_scaled_error_vs_cache':feature_error[0],'maximum_original_parent_probability_error_vs_cache':prob_error[0],'parent_source_sha256':_REGIONAL_PARENT,'parent_checkpoint_sha256':'7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef','regional_helper_sha256':_REGIONAL_SOURCE,'old_global_index_sha256':_REGIONAL_OLD_INDEX,'new_NPZ_payload_bytes':payload_bytes[0],'archive_bytes':archive_path.stat().st_size,'archive_sha256':sha_file(archive_path),'index_bytes':index.stat().st_size,'index_sha256':sha_file(index),'pending_CPU_tasks_cap':4,'GPU_peak_bytes':{d:torch.cuda.max_memory_allocated(d) for d in devices},'elapsed_seconds':time.monotonic()-started,'clinical_labels_reports_read':False,'fitting_or_optimizer':False,'independent_internal_validation':False,'clinical_or_competition_gain':False,'anatomical_localization_proven':False,'competition_submission':False}
    assert archive_path.stat().st_size+index.stat().st_size<128*2**20
    Path('/kaggle/working/regional-features-accepted.json').write_text(json.dumps(receipt,sort_keys=True,indent=2));assert not Path('/kaggle/working/submission.csv').exists();print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    _regional_probe_main()
"""
    regional_sha = hashlib.sha256(regional.encode()).hexdigest()
    for key, value in {
        "ROSTER_VALUE": roster,
        "SOURCE_VALUE": regional_sha,
        "PARENT_VALUE": PARENT_SOURCE_SHA256,
    }.items():
        patch = patch.replace(key, repr(value))
    text = (
        parent[: parent.index(GUARD)]
        + "\nfrom pathlib import Path\n"
        + bodies(context)
        + "\n"
        + bodies(regional)
        + "\n"
        + bodies(bounded)
        + "\n"
        + patch
    )
    compile(text, "licensed_regional_features_probe", "exec")
    return text, {
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_bytes": len(text.encode()),
        "regional_helper_sha256": regional_sha,
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "roster_sha256": hashlib.sha256(
            json.dumps(roster, separators=(",", ":")).encode()
        ).hexdigest(),
        "studies": 32,
        "representation": "same_forward_global_regional_v1",
        "same_forward_global_regional_and_parent_logits": True,
        "new_feature_cap_bytes": 128 * 2**20,
        "clinical_labels_reports_read": False,
        "competition_submission": False,
    }
