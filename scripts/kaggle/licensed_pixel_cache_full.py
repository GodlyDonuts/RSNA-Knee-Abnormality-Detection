"""Four bounded CPU shards of the exact licensed-parent fitting inputs.

Reuses the actual accepted32 pixel members, encodes only new studies, and reads
no labels/reports. The source population comes from its pinned private index.
"""

from __future__ import annotations

import hashlib

from scripts.kaggle.licensed_pixel_cache_probe import build as probe_build

FULL_INDEX_SHA = "5a1a4c77be60a46973c9513b7115ba68edfc7bf76c6205a89de61217a75bf113"
PROBE_INDEX_SHA = "3a1567989bf65779aeb19785dafdedef7386fb6bedba5ec45da6d3a67a9a79e3"
POPULATION_SHA = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
SHARDS = 4
SHARD_SIZE = 972


def shard_bounds(shard):
    if type(shard) is not int or not 0 <= shard < SHARDS:
        raise ValueError("One of four fixed source shards required")
    return shard * SHARD_SIZE, min(3887, (shard + 1) * SHARD_SIZE)


def build(parent, codec, shard):
    lo, hi = shard_bounds(shard)
    base, binding = probe_build(parent, codec)
    guard = "if __name__=='__main__':\n    _pixel_cache_main()\n"
    if not base.endswith(guard) or base.count(guard) != 1:
        raise ValueError("Unique original CPU probe entry point required")
    patch = r"""
_FULL_PIXEL_SHARD = SHARD_VALUE
_FULL_PIXEL_LO = LO_VALUE
_FULL_PIXEL_HI = HI_VALUE
_FULL_PIXEL_INDEX = FULL_INDEX_VALUE
_FULL_PIXEL_PROBE_INDEX = PROBE_INDEX_VALUE
_FULL_PIXEL_UID_SHA = UID_VALUE

def _pixel_find_exact_index(size,sha):
    found=[]
    for directory in Path('/kaggle/input').iterdir():
        if not directory.is_dir() or 'competition' in directory.name or 'abnormality-detection' in directory.name:continue
        if any((directory/n).exists() for n in ('train.csv','test.csv','train_images','train_series')):continue
        for root,dirs,files in os.walk(directory):
            if len(Path(root).relative_to(directory).parts)>=5:dirs[:]=[]
            if 'index.private.json' in files:
                p=Path(root)/'index.private.json'
                if p.stat().st_size==size and hashlib.sha256(p.read_bytes()).hexdigest()==sha:found.append(p)
    assert found,'Exact original private fitting/probe index required'
    return found[0]

def _pixel_full_main():
    import pandas as pd
    assert torch.cuda.device_count()==0 and not Path('/kaggle/working/submission.csv').exists()
    assert (MAXS,IMG,INPUT_RES,CROP_MM)==(96,384,320,140.0) and SLOT_SIZES==[26,22,18,12,18]
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    started=time.monotonic();working=Path('/kaggle/working');free=shutil.disk_usage(working).free;cap=14*2**30
    assert free>cap+2*2**30,'FourteenGiB shard plus2GiB scratch reserve required'
    original_path=_pixel_find_exact_index(1261832,_FULL_PIXEL_INDEX);original=json.loads(original_path.read_text());ids=[r['study_uid'] for r in original['records']]
    assert len(ids)==3887 and ids==sorted(set(ids)) and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==_FULL_PIXEL_UID_SHA
    wanted=ids[_FULL_PIXEL_LO:_FULL_PIXEL_HI];assert len(wanted)<=972
    probe_path=_pixel_find_exact_index(21324,_FULL_PIXEL_PROBE_INDEX);probe=json.loads(probe_path.read_text());old={r['study_uid']:r for r in probe['records']};assert len(old)==32 and set(old)<=set(ids) and probe['codec_sha256']==_CACHE_CODEC_SHA and probe['parent_source_sha256']==_CACHE_PARENT_SHA
    old_tar=tarfile.open(probe_path.parent/'pixels.tar','r:');old_members={m.name:m for m in old_tar.getmembers()};assert len(old_members)==32
    root=Path(find_test_root());series=pd.read_csv(root/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str});series=series.loc[series.StudyInstanceUID.isin(wanted)];assert set(series.StudyInstanceUID)==set(wanted)
    ser={k:v.to_dict('records') for k,v in series.groupby('StudyInstanceUID')};images=root/('train_series' if (root/'train_series').is_dir() else 'train_images');assert images.is_dir();_pre_init()
    out=working/'licensed-parent-pixels-full';out.mkdir(mode=0o700);archive_path=out/'pixels.tar';records=[];payload=0;reused=new=empty=errors=0;elapsed=[]
    with tarfile.open(archive_path,'x:') as archive:
        for i,uid in enumerate(wanted):
            step=time.monotonic()
            if uid in old:
                row=dict(old[uid]);member=old_members[row['file']];assert member.isreg() and member.size==row['bytes'];handle=old_tar.extractfile(member);assert handle is not None;raw=handle.read();volume,mask=decode(raw,row['sha256']);assert hashlib.sha256(volume.tobytes()).hexdigest()==row['volume_sha256'] and hashlib.sha256(mask.tobytes()).hexdigest()==row['mask_sha256'];row['source_key']='probe32';reused+=1
                # Original32 already has complete exact two-view tensor proofs.
                # No MRI redecode, re-encoding, CNN, or new copy of its bytes.
            else:
                failure=None
                try:volume,mask=_READER(uid,ser,str(images));validate(volume,mask)
                except Exception as exc:
                    failure=type(exc).__name__;volume=np.zeros((96,384,384),np.uint8);mask=np.zeros(96,np.uint8);errors+=1
                raw=encode(volume,mask,remaining_bytes=cap-payload-2**20);h=hashlib.sha256(raw).hexdigest();restored,present=decode(raw,h);assert np.array_equal(volume,restored) and np.array_equal(mask,present)
                hashes=[]
                for mirror in (False,True):
                    original_x=eval_windows(volume,mask,94,320,norm=NORM,mirror=mirror);cached_x=eval_windows(restored,present,94,320,norm=NORM,mirror=mirror);assert torch.equal(original_x,cached_x);hashes.append(_pixel_tensor_sha(original_x));del original_x,cached_x
                name=hashlib.sha256(uid.encode()).hexdigest()+'.npz';info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600;archive.addfile(info,io.BytesIO(raw));payload+=len(raw);new+=1
                row={'study_uid':uid,'file':name,'archive':'pixels.tar','source_key':'self','bytes':len(raw),'sha256':h,'volume_sha256':hashlib.sha256(volume.tobytes()).hexdigest(),'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'original_plain_input_sha256':hashes[0],'original_mirrored_input_sha256':hashes[1],'preprocessing_error_type':failure}
            row['has_original_MRI_pixels']=bool(mask.any());empty+=int(not row['has_original_MRI_pixels']);records.append(row);elapsed.append(time.monotonic()-step);assert time.monotonic()-started<18000
            if len(records)%32==0:print(json.dumps({'pixel_shard':_FULL_PIXEL_SHARD,'completed':len(records),'total':len(wanted),'new_cases':new,'original32_reused':reused,'empty':empty,'decode_errors':errors,'elapsed_seconds':time.monotonic()-started}),flush=True)
    old_tar.close();assert [r['study_uid'] for r in records]==wanted and len(records)==new+reused
    index=out/'index.private.json';index.write_text(json.dumps({'version':VERSION,'kind':'licensed_parent_full_fitting_pixel_cache_shard_v1','records':records,'shard':_FULL_PIXEL_SHARD,'shards':4,'lo':_FULL_PIXEL_LO,'hi':_FULL_PIXEL_HI,'global_fitting_studies':3887,'global_uid_sha256':_FULL_PIXEL_UID_SHA,'original_full_feature_index_sha256':_FULL_PIXEL_INDEX,'original_probe_pixel_index_sha256':_FULL_PIXEL_PROBE_INDEX,'codec_sha256':_CACHE_CODEC_SHA,'parent_source_sha256':_CACHE_PARENT_SHA,'reports_clinical_labels_protected_values_read':False},sort_keys=True));index.chmod(0o600)
    assert archive_path.stat().st_size+index.stat().st_size<=cap
    def digest(path):
        with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    receipt={'status':'actual_licensed_parent_full_fitting_pixel_shard_complete','shard':_FULL_PIXEL_SHARD,'shards':4,'lo':_FULL_PIXEL_LO,'hi':_FULL_PIXEL_HI,'studies':len(records),'global_fitting_studies':3887,'original32_reused':reused,'new_original_MRI_decodes':new,'empty_studies':empty,'decode_errors':errors,'GPU_fitting_admitted':False,'all_new_original_volume_mask_and_both_view_tensors_bit_equal_after_roundtrip':True,'original32_reused_without_decode_reencode_or_new_archive_copy':True,'archive_bytes':archive_path.stat().st_size,'archive_sha256':digest(archive_path),'index_bytes':index.stat().st_size,'index_sha256':digest(index),'NPZ_payload_bytes':payload,'global_uid_sha256':_FULL_PIXEL_UID_SHA,'parent_source_sha256':_CACHE_PARENT_SHA,'codec_sha256':_CACHE_CODEC_SHA,'original_full_index_sha256':_FULL_PIXEL_INDEX,'original_probe_pixel_index_sha256':_FULL_PIXEL_PROBE_INDEX,'CPU_affinity':len(os.sched_getaffinity(0)),'working_free_bytes_before':free,'output_cap_bytes':cap,'maximum_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'elapsed_seconds':time.monotonic()-started,'mean_case_seconds':float(np.mean(elapsed)),'reports_clinical_labels_protected_values_read':False,'new_weight_download_or_encoder_optimization':False,'clinical_or_competition_accuracy_claim':False,'competition_entry':False}
    (working/'licensed-pixels-full-shard-accepted.json').write_text(json.dumps(receipt,sort_keys=True,indent=2));assert not (working/'submission.csv').exists();print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    _pixel_full_main()
"""
    for name, value in {
        "SHARD_VALUE": shard,
        "LO_VALUE": lo,
        "HI_VALUE": hi,
        "FULL_INDEX_VALUE": FULL_INDEX_SHA,
        "PROBE_INDEX_VALUE": PROBE_INDEX_SHA,
        "UID_VALUE": POPULATION_SHA,
    }.items():
        patch = patch.replace(name, repr(value))
    source = base[: -len(guard)] + patch
    compile(source, "licensed_parent_full_CPU_pixel_shard", "exec")
    return source, {
        **binding,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_bytes": len(source.encode()),
        "shard": shard,
        "lo": lo,
        "hi": hi,
        "fitting_studies": hi - lo,
        "global_fitting_studies": 3887,
        "maximum_output_bytes": 14 * 2**30,
        "global_uid_sha256": POPULATION_SHA,
        "full_original_index_sha256": FULL_INDEX_SHA,
        "original_probe_pixel_index_sha256": PROBE_INDEX_SHA,
        "actual_GPU_fitting_admission": False,
    }
