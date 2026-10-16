"""Expand the accepted coherent regional cache, reusing its original32 records."""

from __future__ import annotations

import hashlib
import json

from scripts.kaggle.regional_feature_probe import _build as probe_source
from scripts.kaggle.series_boundary_notebook import PARENT_SOURCE_SHA256

ROSTER_SHA = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
FIRST32_SHA = "6ca592004ae972a93a8bb96dc7024b46c7ad86525388c02542dca844a6b1605f"
REUSE_REF = "godlydonuts/rsna-regional-features-511654eced21"
REUSE_INDEX_SHA = "280b3d04e4506a1fc0771280a16a6c1a9fe6ae3455ffdac8fbbc048f2a343481"
REUSE_ARCHIVE_SHA = "cc2f06ec819a707bc72c55c0a16778493d428a450130b3f8abb0bc6eee0aedee"
NEW_ARCHIVE_CAP = 14 * 2**30


def uid_hash(roster):
    return hashlib.sha256(json.dumps(roster, separators=(",", ":")).encode()).hexdigest()


def build(parent, context, regional, bounded, roster):
    if (
        hashlib.sha256(parent.encode()).hexdigest() != PARENT_SOURCE_SHA256
        or uid_hash(roster) != ROSTER_SHA
        or uid_hash(roster[:32]) != FIRST32_SHA
    ):
        raise ValueError("Exact admitted parent and3887fitting-only roster required")
    return _build(parent, context, regional, bounded, roster)


def _build(parent, context, regional, bounded, roster):
    if len(roster) != 3887 or roster != sorted(set(roster)):
        raise ValueError("Exactly3887distinct sorted fitting-only identities required")
    text, _ = probe_source(parent, context, regional, bounded, roster[:32])

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError("Accepted original32 compiler contract changed")
        text = text.replace(old, new)

    replace("_REGIONAL_ROSTER = " + repr(roster[:32]), "_REGIONAL_ROSTER = " + repr(roster))
    replace(
        "torch.set_num_threads(1);old=_regional_old_cache()",
        "torch.set_num_threads(1);reused=_regional_reuse32()",
    )
    replace(
        "records=[];failures=[];qualified={};payload_bytes=[0];feature_error=[0.];prob_error=[0.];old_exact=[0]",
        "records=list(reused);failures=[];qualified={};payload_bytes=[0]",
    )
    replace(
        "assert np.array_equal(cs,old[sid]['centers']) and np.array_equal(mask[cs].astype(bool),old[sid]['present'])",
        "assert cs.shape==(94,) and mask[cs].any()",
    )
    start = text.index("                        stage='old_probability_reference';")
    stop = text.index("                        local.append(regions.cpu().numpy())", start)
    text = (
        text[:start]
        + "                        actual=f[0].float().cpu().numpy();assert np.isfinite(actual).all()\n"
        + text[stop:]
    )
    replace("payload_bytes[0]<128*2**20", "payload_bytes[0]<14*2**30")
    replace(
        "'archive':'features.tar','bytes':len(raw)",
        "'archive':'features.tar','source_key':'self','bytes':len(raw)",
    )
    replace("len(records)%8==0", "len(records)%128==0")
    replace("'total':32", "'total':3887")
    replace(
        "bounded_results(pool,_regional_pre_one,_REGIONAL_ROSTER,limit=4)",
        "bounded_results(pool,_regional_pre_one,_REGIONAL_ROSTER[32:],limit=4)",
    )
    replace("time.monotonic()-started<2700", "time.monotonic()-started<10800")
    text = text.replace("len(records)==32", "len(records)==3887")
    start = text.index("    receipt={'status':")
    stop = text.index("\n    assert archive_path.stat()", start)
    text = (
        text[:start]
        + r"""    receipt={'status':'actual3887_same_forward_regional_features_complete','studies':3887,'reused32_studies':32,'new_CNN_studies':3855,'roster_sha256':hashlib.sha256(json.dumps(_REGIONAL_ROSTER,separators=(',',':')).encode()).hexdigest(),'views':2,'windows_per_view':94,'region_summary_count':7,'channels_retained':1024,'regional_storage_dtype':'float16','global_feature_dtype':'float32','same_forward_global_regional_and_parent_logits':True,'reused32_index_sha256':_REGIONAL_REUSE_INDEX_SHA,'reused32_archive_sha256':_REGIONAL_REUSE_ARCHIVE_SHA,'parent_source_sha256':_REGIONAL_PARENT,'parent_checkpoint_sha256':'7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef','regional_helper_sha256':_REGIONAL_SOURCE,'devices':list(qualified.values()),'new_NPZ_payload_bytes':payload_bytes[0],'archive_bytes':archive_path.stat().st_size,'archive_sha256':sha_file(archive_path),'index_bytes':index.stat().st_size,'index_sha256':sha_file(index),'GPU_peak_bytes':{d:torch.cuda.max_memory_allocated(d) for d in devices},'elapsed_seconds':time.monotonic()-started,'clinical_labels_reports_read':False,'fitting_or_optimizer':False,'independent_internal_validation':False,'clinical_or_competition_gain':False,'competition_submission':False}"""
        + text[stop:]
    )
    replace(
        "'old_global_index_sha256':_REGIONAL_OLD_INDEX",
        "'reused32_index_sha256':_REGIONAL_REUSE_INDEX_SHA",
    )
    replace(
        "archive_path.stat().st_size+index.stat().st_size<128*2**20",
        "archive_path.stat().st_size+index.stat().st_size<14*2**30",
    )
    text = text.replace("released-regional-features'", "released-regional-features-full'")
    text = text.replace("regional-features-accepted.json", "regional-full-features-accepted.json")
    text = text.replace("regional-probe-observation.json", "regional-full-observation.json")

    reuse = r"""
_REGIONAL_REUSE_INDEX_SHA = INDEX_SHA_VALUE
_REGIONAL_REUSE_ARCHIVE_SHA = ARCHIVE_SHA_VALUE
def _regional_reuse32():
    import io,tarfile
    candidates=[]
    for d in Path('/kaggle/input').iterdir():
        if not d.is_dir() or 'competition' in d.name or 'abnormality-detection' in d.name:continue
        if any((d/x).exists() for x in ('train.csv','test.csv','train_images','train_series')):continue
        for root,dirs,files in os.walk(d):
            if len(Path(root).relative_to(d).parts)>=5:dirs[:]=[]
            if 'index.private.json' in files:
                p=Path(root)/'index.private.json'
                if p.stat().st_size==9787 and hashlib.sha256(p.read_bytes()).hexdigest()==_REGIONAL_REUSE_INDEX_SHA:candidates.append(p)
    assert candidates;path=candidates[0];value=json.loads(path.read_text());assert value['representation']=='same_forward_global_regional_v1' and [r['study_uid'] for r in value['records']]==_REGIONAL_ROSTER[:32]
    archive=path.parent/'features.tar';assert archive.stat().st_size==111052800
    with archive.open('rb') as handle:assert hashlib.file_digest(handle,'sha256').hexdigest()==_REGIONAL_REUSE_ARCHIVE_SHA
    by_name={r['file']:r for r in value['records']};seen=set()
    with tarfile.open(archive,'r:') as handle:
        for m in handle:
            assert m.isfile() and m.name in by_name and m.name not in seen;row=by_name[m.name];assert m.size==row['bytes'] and m.size<4*2**20;raw=handle.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==row['sha256'];seen.add(m.name)
            with np.load(io.BytesIO(raw),allow_pickle=False) as a:
                assert a['features'].shape==(2,94,1024) and a['features'].dtype==np.float32 and np.isfinite(a['features']).all()
                assert a['regional'].shape==(2,94,7,1024) and a['regional'].dtype==np.float16 and np.isfinite(a['regional']).all()
                assert a['parent_logits'].shape==(2,12) and a['parent_logits'].dtype==np.float32 and np.isfinite(a['parent_logits']).all()
    assert seen==set(by_name)
    return [dict(r,source_key='regional32') for r in value['records']]

""".replace("INDEX_SHA_VALUE", repr(REUSE_INDEX_SHA)).replace(
        "ARCHIVE_SHA_VALUE", repr(REUSE_ARCHIVE_SHA)
    )
    guard = "if __name__=='__main__':"
    replace(guard, reuse + guard)
    compile(text, "full_coherent_regional_features", "exec")
    return text, {
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_bytes": len(text.encode()),
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "regional_helper_sha256": hashlib.sha256(regional.encode()).hexdigest(),
        "studies": 3887,
        "reused_studies": 32,
        "new_CNN_studies": 3855,
        "roster_sha256": uid_hash(roster),
        "representation": "same_forward_global_regional_v1",
        "new_archive_cap_bytes": NEW_ARCHIVE_CAP,
        "reused_ref": REUSE_REF,
        "reused_index_sha256": REUSE_INDEX_SHA,
        "clinical_labels_reports_read": False,
        "competition_submission": False,
    }
