"""Compile a private CPU32 lossless-input gate from the exact scored parent.

External implementation and private identities are supplied only on authorized
storage, never vendored here. No CNN, report, optimizer or competition CSV path.
"""

from __future__ import annotations

import ast
import hashlib

from scripts.kaggle.released_context_notebook import bodies, segment
from scripts.kaggle.series_boundary_notebook import PARENT_SOURCE_SHA256

CODEC_SHA = "7bdaf41a864b499401ff4a62e6288d30eead3849c0fda5c49cc67b3b2d0e2ab5"
CONSTANTS = {
    "INPUT_RES",
    "INPUT_MODE",
    "IMG",
    "CROP_MM",
    "SLOTS",
    "MAXS",
    "HDR_THREADS",
    "PX_THREADS",
    "NORM",
    "FASTREAD_VARIANT",
    "FR_HDR",
    "FR_PX",
    "_FASTREAD_B64",
    "_MEAN",
    "_STD",
    "SLOT_SIZES",
    "SAGITTAL_SLOTS",
}
FUNCTIONS = {
    "_load_fastread",
    "_make_reader",
    "_pick_series_for_slot",
    "build_study",
    "_pre_init",
    "_eval_centers",
    "_slot_index",
    "_mirror_tri",
    "eval_windows",
    "find_test_root",
}


def build(parent, codec):
    if hashlib.sha256(parent.encode()).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Exact licensed scored MRI parent source required")
    return _build(parent, codec)


def _build(parent, codec):
    if hashlib.sha256(codec.encode()).hexdigest() != CODEC_SHA:
        raise ValueError("Exact bounded lossless cache codec required")
    pieces = []
    seen: set[str] = set()
    for node in ast.parse(parent).body:
        names = set()
        if isinstance(node, ast.Assign):
            names = {
                n.id for target in node.targets for n in ast.walk(target) if isinstance(n, ast.Name)
            }
            if not names <= CONSTANTS:
                continue
        elif isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            names = {node.name}
        if names & (CONSTANTS | FUNCTIONS):
            if seen & names:
                raise ValueError("Unique original preprocessing definitions required")
            pieces.append(segment(parent, node))
            seen |= names
    if seen != CONSTANTS | FUNCTIONS:
        raise ValueError("Complete original preprocessing and CPU window renderer required")
    imports = """
import os,glob,time,json,io,hashlib,resource,tarfile,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.nn.functional as F
"""
    patch = r"""
_CACHE_PARENT_SHA = PARENT_VALUE
_CACHE_CODEC_SHA = CODEC_VALUE
_CACHE_OLD_INDEX = 'a1bba9e618444f16ddd453efa1bb8650245f8259dcd9a00348b17efd87c1e100'

def _pixel_original32():
    candidates=[]
    for directory in Path('/kaggle/input').iterdir():
        if not directory.is_dir() or 'competition' in directory.name or 'abnormality-detection' in directory.name:continue
        if any((directory/n).exists() for n in ('train.csv','test.csv','train_images','train_series')):continue
        for root,dirs,files in os.walk(directory):
            if len(Path(root).relative_to(directory).parts)>=5:dirs[:]=[]
            if 'index.private.json' in files:
                p=Path(root)/'index.private.json'
                if p.stat().st_size==8574 and hashlib.sha256(p.read_bytes()).hexdigest()==_CACHE_OLD_INDEX:candidates.append(p)
    assert candidates,'Original private admitted32 feature index required'
    p=candidates[0];value=json.loads(p.read_text());rows=value['records'];assert len(rows)==32 and len({r['study_uid'] for r in rows})==32
    assert value['weights_sha256']=='7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef'
    return p,rows

def _pixel_tensor_sha(value):
    assert value.dtype==torch.float32 and value.shape==(94,3,320,320)
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()

def _pixel_cache_main():
    import pandas as pd
    assert torch.cuda.device_count()==0,'This input gate is CPU only'
    assert not Path('/kaggle/working/submission.csv').exists()
    assert (MAXS,IMG,INPUT_RES,CROP_MM)==(96,384,320,140.0) and SLOT_SIZES==[26,22,18,12,18]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started=time.monotonic();working=Path('/kaggle/working');free=shutil.disk_usage(working).free
    assert free>3*2**30,'Bounded1GiB output plus2GiB scratch reserve required'
    index_path,rows=_pixel_original32();ids=[r['study_uid'] for r in rows]
    root=Path(find_test_root());series=pd.read_csv(root/'train_series.csv',dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    series=series.loc[series.StudyInstanceUID.isin(ids)];assert set(series.StudyInstanceUID)==set(ids)
    records_by_study={k:v.to_dict('records') for k,v in series.groupby('StudyInstanceUID')}
    image_root=root/('train_series' if (root/'train_series').is_dir() else 'train_images');assert image_root.is_dir()
    _pre_init();slow_reader=_make_reader();directory=working/'licensed-parent-pixels32';directory.mkdir(mode=0o700)
    archive_path=directory/'pixels.tar';records=[];payload_bytes=0;timings=[];sizes=[];slow=[]
    with tarfile.open(archive_path,'x:') as archive:
        for i,row in enumerate(rows):
            step=time.monotonic();uid=row['study_uid'];volume,mask=_READER(uid,records_by_study,str(image_root));validate(volume,mask)
            assert mask.any(),'A failed or empty decoded MRI is not cache admission'
            cs=np.asarray([max(1,min(c,94)) for c in _eval_centers(mask,96,94)],dtype=np.int64)
            old_path=index_path.parent/row['file'];old_raw=old_path.read_bytes();assert len(old_raw)==row['bytes'] and hashlib.sha256(old_raw).hexdigest()==row['sha256']
            with np.load(io.BytesIO(old_raw),allow_pickle=False) as old:
                assert np.array_equal(cs,old['centers']) and np.array_equal(mask[cs].astype(bool),old['present'])
            raw=encode(volume,mask,remaining_bytes=512*2**20-payload_bytes);raw_sha=hashlib.sha256(raw).hexdigest();restored,present=decode(raw,raw_sha)
            assert np.array_equal(volume,restored) and np.array_equal(mask,present)
            hashes=[]
            for mirror in (False,True):
                original=eval_windows(volume,mask,94,320,norm=NORM,mirror=mirror)
                cached=eval_windows(restored,present,94,320,norm=NORM,mirror=mirror)
                assert torch.equal(original,cached),'Lossless input tensors changed'
                hashes.append(_pixel_tensor_sha(original));del original,cached
            if i<2:
                reference,reference_mask=build_study(uid,records_by_study,str(image_root),slow_reader)
                slow.append({'different_volume_pixels':int(np.count_nonzero(reference!=volume)),'maximum_uint8_difference':int(np.abs(reference.astype(np.int16)-volume.astype(np.int16)).max()),'mask_equal':bool(np.array_equal(reference_mask,mask))})
                del reference,reference_mask
            name=hashlib.sha256(uid.encode()).hexdigest()+'.npz';info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o600;archive.addfile(info,io.BytesIO(raw));payload_bytes+=len(raw)
            records.append({'study_uid':uid,'file':name,'archive':'pixels.tar','bytes':len(raw),'sha256':raw_sha,'volume_sha256':hashlib.sha256(volume.tobytes()).hexdigest(),'mask_sha256':hashlib.sha256(mask.tobytes()).hexdigest(),'original_plain_input_sha256':hashes[0],'original_mirrored_input_sha256':hashes[1]})
            timings.append(time.monotonic()-step);sizes.append(len(raw));assert time.monotonic()-started<3000
            if len(records)%8==0:print(json.dumps({'original_pixel_studies_cached':len(records),'total':32}),flush=True)
    index=directory/'index.private.json';index.write_text(json.dumps({'version':VERSION,'records':records,'parent_source_sha256':_CACHE_PARENT_SHA,'codec_sha256':_CACHE_CODEC_SHA,'original_admitted_feature_index_sha256':_CACHE_OLD_INDEX,'reports_clinical_labels_protected_values_read':False},sort_keys=True));index.chmod(0o600)
    assert archive_path.stat().st_size+index.stat().st_size<512*2**20
    def digest(path):
        with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    receipt={'status':'actual32_licensed_parent_CPU_lossless_pixels_and_all94_two_view_inputs_accepted','studies':32,'views':2,'windows_per_view':94,'volume_shape':[96,384,384],'volume_dtype':'uint8','mask_dtype':'uint8','rendered_input_shape':[94,3,320,320],'rendered_input_dtype':'float32','all_original_volume_mask_and_both_view_tensors_bit_equal_after_roundtrip':True,'all_original_admitted_center_metadata_equal':True,'archive_bytes':archive_path.stat().st_size,'archive_sha256':digest(archive_path),'index_bytes':index.stat().st_size,'index_sha256':digest(index),'NPZ_payload_bytes':payload_bytes,'original_uncompressed_pixel_mask_bytes':32*(96*384*384+96),'compressed_ratio':payload_bytes/(32*(96*384*384+96)),'median_member_bytes':float(np.median(sizes)),'maximum_member_bytes':max(sizes),'per_study_seconds_mean':float(np.mean(timings)),'per_study_seconds_max':max(timings),'slow_original_reference_secondary_two_studies':slow,'CPU_affinity':len(os.sched_getaffinity(0)),'working_free_bytes_before':free,'maximum_process_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'elapsed_seconds':time.monotonic()-started,'parent_source_sha256':_CACHE_PARENT_SHA,'codec_sha256':_CACHE_CODEC_SHA,'original_feature_index_sha256':_CACHE_OLD_INDEX,'new_weight_download_or_encoder_optimization':False,'reports_clinical_labels_protected_values_read':False,'independent_internal_validation':False,'clinical_or_competition_accuracy_claim':False,'competition_entry':False,'full3887_cache':False}
    (working/'licensed-pixels32-accepted.json').write_text(json.dumps(receipt,sort_keys=True,indent=2));assert not (working/'submission.csv').exists();print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    _pixel_cache_main()
"""
    patch = patch.replace("PARENT_VALUE", repr(PARENT_SOURCE_SHA256)).replace(
        "CODEC_VALUE", repr(CODEC_SHA)
    )
    source = imports + "\n".join(pieces) + "\n" + bodies(codec) + "\n" + patch
    compile(source, "licensed_parent_CPU_pixel_cache", "exec")
    return source, {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_bytes": len(source.encode()),
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "codec_sha256": CODEC_SHA,
        "CPU_only": True,
        "fitting_studies": 32,
        "maximum_output_bytes": 512 * 2**20,
        "no_model_report_optimizer_or_competition_CSV_path": True,
    }
