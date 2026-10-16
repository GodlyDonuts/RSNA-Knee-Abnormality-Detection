"""Build a private, label-free feature/zero-start probe from the licensed parent.

The external source is supplied from its original ignored receipt and is never
vendored in Git. This probe is not training and cannot produce submission.csv.
"""

from __future__ import annotations

import ast
import hashlib
import json

from scripts.kaggle.series_boundary_notebook import GUARD, PARENT_SOURCE_SHA256


def build(parent, helper, roster):
    if hashlib.sha256(parent.encode()).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Exact previously scored licensed parent required")
    return _build(parent, helper, roster)


def _build(parent, helper, roster):
    if (
        not isinstance(roster, list)
        or len(roster) != 32
        or len(set(roster)) != 32
        or not all(isinstance(x, str) and x for x in roster)
        or parent.count(GUARD) != 1
        or parent[parent.index(GUARD) + len(GUARD) :].strip()
    ):
        raise ValueError("Distinct fixed 32 engineering studies and unique parent guard required")
    body = []
    for node in ast.parse(helper).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        body.append(ast.get_source_segment(helper, node))
    patch = r"""
_CONTEXT_ROSTER = ROSTER_VALUE
_CONTEXT_HELPER_SHA = HELPER_SHA_VALUE
_CONTEXT_PARENT_SHA = PARENT_SHA_VALUE

def _context_private_pre_one(sid):
    # No case identifiers or raw exception text in stdout. Detailed data remains
    # in the private feature index only; no labels/reports are read.
    try:
        vol, mask = _READER(sid, _SER, _TSDIR)
        return sid, vol, mask
    except Exception:
        return sid, None, None

def _context_probe_main():
    import multiprocessing as mp
    import pandas as pd
    started = time.monotonic()
    assert not Path('/kaggle/working/submission.csv').exists()
    assert torch.cuda.device_count() == 2 and K_EVAL == 94 and INPUT_RES == 320
    assert all('T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert SLOT_SIZES == list(SIZES) and len(ARMS) == 1
    torch.set_num_threads(1)
    out = Path('/kaggle/working/released-context-features')
    out.mkdir(mode=0o700)
    root = Path(find_test_root())
    series = pd.read_csv(root/'train_series.csv', dtype={'StudyInstanceUID':str,'SeriesInstanceUID':str})
    series = series.loc[series.StudyInstanceUID.isin(_CONTEXT_ROSTER)]
    assert set(series.StudyInstanceUID) == set(_CONTEXT_ROSTER)
    global _SER, _TSDIR, _READER
    _SER = {k:v.to_dict('records') for k,v in series.groupby('StudyInstanceUID')}
    _TSDIR = str(root/('train_series' if (root/'train_series').is_dir() else 'train_images'))
    assert Path(_TSDIR).is_dir()
    devices = ['cuda:0','cuda:1']
    models = {d:load_model(find_weight_file(ARMS[0]['file']),ARMS[0]['arch'],ARMS[0]['res'],d)[0] for d in devices}
    contexts = {d:ContextResidual().eval().to(d) for d in devices}
    assert all(not m.training for model in models.values() for m in model.modules())
    q = queue.Queue(maxsize=2)
    lock = threading.Lock()
    records, failures, qualified = [], [], {}
    emitted = [0]

    def consumer(device):
        with torch.cuda.device(device), torch.no_grad():
            model, context = models[device], contexts[device]
            torch.cuda.reset_peak_memory_stats(device)
            while True:
                item = q.get()
                try:
                    if item is None:
                        return
                    sid,vol,mask = item
                    assert vol is not None and mask is not None and mask.sum()>0
                    cs = np.asarray([max(1,min(c,94)) for c in _eval_centers(mask,96,94)], dtype=np.int64)
                    centers = torch.from_numpy(cs)
                    slots, depth = window_metadata(centers)
                    other, mirrored_depth = window_metadata(centers,mirrored=True)
                    assert torch.equal(slots,other)
                    features, logits = [], []
                    for mirror in (False,True):
                        x = gpu_windows(vol,mask,94,320,device,norm=NORM,mirror=mirror)
                        if device not in qualified:
                            reference = eval_windows(vol,mask,94,320,norm=NORM,mirror=mirror)
                            assert torch.equal(x.cpu(),reference)
                        with torch.autocast('cuda',dtype=torch.float16):
                            f = model.encode(x[None])
                            value = model.head(f).float()
                        assert f.shape==(1,94,1024) and f.dtype in (torch.float16,torch.float32)
                        assert value.shape==(1,12) and torch.isfinite(f).all() and torch.isfinite(value).all()
                        frozen = f[0].cpu().numpy()
                        restored = torch.from_numpy(frozen.copy()).to(device)[None]
                        with torch.autocast('cuda',dtype=torch.float16):
                            restored_logit = model.head(restored).float()
                        assert torch.equal(value,restored_logit)
                        if device not in qualified:
                            direct = infer_probs(model,x,device)
                            assert np.array_equal(value.sigmoid()[0].cpu().numpy(),direct)
                        with torch.autocast('cuda',enabled=False):
                            r = mirrored_depth if mirror else depth
                            present = torch.from_numpy(mask[cs].astype(bool))[None].to(device)
                            adjusted,delta = context(f.float(),value,slots[None].to(device),r[None].to(device),present,torch.ones_like(present))
                        assert torch.equal(adjusted,value) and not delta.count_nonzero()
                        features.append(frozen)
                        logits.append(value[0].cpu().numpy())
                    arrays = {'features':np.stack(features),'parent_logits':np.stack(logits),'centers':cs,'slots':slots.numpy(),'depth':np.stack([depth.numpy(),mirrored_depth.numpy()]),'present':mask[cs].astype(bool)}
                    assert arrays['features'].dtype in (np.float16,np.float32)
                    # Bound before writing; even FP32 features across all32 fit.
                    with lock:
                        assert emitted[0] + arrays['features'].nbytes + 65536 <= 32*2**20
                        emitted[0] += arrays['features'].nbytes + 65536
                    filename = hashlib.sha256(sid.encode()).hexdigest()+'.npz'
                    target = out/filename
                    with target.open('xb') as handle:
                        np.savez(handle,**arrays)
                    os.chmod(target,0o600)
                    with target.open('rb') as handle:
                        file_sha = hashlib.file_digest(handle,'sha256').hexdigest()
                    with lock:
                        records.append({'study_uid':sid,'file':filename,'sha256':file_sha,'bytes':target.stat().st_size})
                        qualified[device]={'device':device,'name':torch.cuda.get_device_name(device),'actual_CPU_GPU_original_pixels_exact':True,'cached_head_exact_parent_probability':True,'zero_start_exact_parent_logits':True,'feature_dtype':str(arrays['features'].dtype)}
                        if len(records)%8==0:print(json.dumps({'context_feature_studies_complete':len(records),'total':32}),flush=True)
                except Exception as exc:
                    with lock:failures.append(type(exc).__name__)
                finally:q.task_done()

    threads = [threading.Thread(target=consumer,args=(d,),daemon=True) for d in devices]
    for thread in threads:thread.start()
    with mp.get_context('fork').Pool(2,initializer=_pre_init) as pool:
        for item in pool.imap_unordered(_context_private_pre_one,_CONTEXT_ROSTER,chunksize=1):
            assert time.monotonic()-started<2700
            q.put(item)
    for _ in threads:q.put(None)
    q.join()
    for thread in threads:thread.join()
    assert not failures and len(records)==32 and len(qualified)==2
    records.sort(key=lambda r:_CONTEXT_ROSTER.index(r['study_uid']))
    index = out/'index.private.json'
    index.write_text(json.dumps({'records':records,'parent_source_sha256':_CONTEXT_PARENT_SHA,'helper_sha256':_CONTEXT_HELPER_SHA,'weights_sha256':'7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef','independent_internal_validation':False},sort_keys=True))
    os.chmod(index,0o600)
    receipt = {'status':'actual_released_context_features_and_zero_start_accepted','studies':32,'view_count':2,'windows_per_view':94,'feature_dimension':1024,'devices':list(qualified.values()),'context_parameters':sum(p.numel() for p in contexts[devices[0]].parameters()),'index_sha256':hashlib.sha256(index.read_bytes()).hexdigest(),'index_bytes':index.stat().st_size,'feature_bytes':sum(r['bytes'] for r in records),'parent_source_sha256':_CONTEXT_PARENT_SHA,'helper_sha256':_CONTEXT_HELPER_SHA,'checkpoint_sha256':'7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef','elapsed_seconds':time.monotonic()-started,'GPU_peak_bytes':{d:torch.cuda.max_memory_allocated(d) for d in devices},'clinical_labels_reports_read':False,'fitting_or_optimizer':False,'independent_internal_validation':False,'clinical_or_competition_gain':False,'competition_submission':False}
    Path('/kaggle/working/context-features-accepted.json').write_text(json.dumps(receipt,sort_keys=True,indent=2))
    assert not Path('/kaggle/working/submission.csv').exists()
    print(json.dumps(receipt),flush=True)

if __name__ == '__main__':
    _context_probe_main()
"""
    patch = patch.replace("ROSTER_VALUE", repr(roster))
    patch = patch.replace("HELPER_SHA_VALUE", repr(hashlib.sha256(helper.encode()).hexdigest()))
    patch = patch.replace("PARENT_SHA_VALUE", repr(PARENT_SOURCE_SHA256))
    source = parent[: parent.index(GUARD)] + "\n" + "\n\n".join(body) + patch
    ast.parse(source)
    binding = {
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "source_bytes": len(source.encode()),
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "helper_sha256": hashlib.sha256(helper.encode()).hexdigest(),
        "roster_sha256": hashlib.sha256(
            json.dumps(roster, separators=(",", ":")).encode()
        ).hexdigest(),
        "studies": 32,
        "clinical_labels_reports": False,
        "competition_submission": False,
        "independent_internal_validation": False,
    }
    return source, binding
