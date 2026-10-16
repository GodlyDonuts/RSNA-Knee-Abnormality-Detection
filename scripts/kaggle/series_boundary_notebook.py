"""Adapt the pinned licensed 0.949 reference without changing its learned weights.

The external implementation is read from its existing private source receipt; it
is not vendored here. Only donor indexing/pooling membership and engineering
receipts change. These are competition inference hypotheses, not independent OOF.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

PARENT_SOURCE_SHA256 = "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde"
WEIGHTS_SHA256 = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
POLICIES = {"within-series", "edge-repeat"}
GPU_INDEX = "vol[np.stack([cs - 1, cs, cs + 1], 1)]"
CPU_INDEX = "np.stack([vol[c - 1], vol[c], vol[c + 1]], 0)"
GUARD = '\nif __name__ == "__main__":\n    main()'


def instrument(parent: str, helper: str, policy: str) -> str:
    if hashlib.sha256(parent.encode()).hexdigest() != PARENT_SOURCE_SHA256:
        raise ValueError("Exact previously scored external source required")
    return _instrument(parent, helper, policy)


def _instrument(parent: str, helper: str, policy: str) -> str:
    if policy not in POLICIES or any(parent.count(n) != 1 for n in (GPU_INDEX, CPU_INDEX, GUARD)):
        raise ValueError("Known boundary policy and unique original transformation sites required")
    if parent[parent.index(GUARD) + len(GUARD) :].strip():
        raise ValueError("Unexpected executable suffix after original main")
    parts = []
    for node in ast.parse(helper).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        parts.append(ast.get_source_segment(helper, node))
    helper_body = "\n\n".join(parts)
    source = parent[: parent.index(GUARD)]
    source = source.replace(GPU_INDEX, "vol[_boundary_source_indices(cs, D)]")
    source = source.replace(CPU_INDEX, "vol[_boundary_source_indices(np.asarray([c]), D)[0]]")
    patch = r"""
# Label-free boundary experiment over the attributed frozen reference.
_BOUNDARY_POLICY = BOUNDARY_POLICY_VALUE
_BOUNDARY_PARENT_SHA = PARENT_SHA_VALUE
_BOUNDARY_HELPER_SHA = HELPER_SHA_VALUE
_BOUNDARY_WEIGHTS_SHA = WEIGHTS_SHA_VALUE
_boundary_records = []
_boundary_prediction_counts = {'started': 0, 'completed': 0, 'failed': 0}
_boundary_lock = threading.Lock()

def _boundary_source_indices(cs, D):
    assert D == sum(SLOT_SIZES) == 96
    mode = 'edge-repeat' if _BOUNDARY_POLICY == 'edge-repeat' else 'legacy'
    return series_donor_indices(cs, SLOT_SIZES, mode=mode)

def _boundary_centers(vol, mask, k):
    D = len(vol)
    return np.asarray([max(1, min(c, D - 2)) for c in _eval_centers(mask, D, k)], dtype=np.int64)

def _boundary_keep(cs):
    keep = same_series_windows(cs, SLOT_SIZES)
    # If every repeated candidate falls on one join, retain the original bag;
    # this is explicit fallback, never an empty study or fabricated evidence.
    fallback = not bool(keep.any())
    return (np.ones(len(cs), dtype=bool) if fallback or _BOUNDARY_POLICY != 'within-series' else keep), fallback

_boundary_original_gpu_windows = gpu_windows
_boundary_original_eval_windows = eval_windows
_boundary_original_infer_probs = infer_probs

def eval_windows(vol, mask, k, res, norm=NORM, mirror=False):
    values = _boundary_original_eval_windows(vol, mask, k, res, norm=norm, mirror=mirror)
    keep, _ = _boundary_keep(_boundary_centers(vol, mask, k))
    return values[keep]

def gpu_windows(vol, mask, k, res, dev, norm=NORM, mirror=False):
    values = _boundary_original_gpu_windows(vol, mask, k, res, dev, norm=norm, mirror=mirror)
    centers = _boundary_centers(vol, mask, k)
    keep, fallback = _boundary_keep(centers)
    mixed = ~same_series_windows(centers, SLOT_SIZES)
    result = values[torch.as_tensor(keep, device=values.device)]
    if not (_boundary_qualifying[0]):
        with _boundary_lock:
            _boundary_records.append({'device':str(dev),'original_windows':len(values),'retained_windows':len(result),'mixed_series_windows':int(mixed.sum()),'all_crossing_fallback':fallback,'mirror':bool(mirror)})
    return result

def infer_probs(model, xwins, device):
    with _boundary_lock:
        _boundary_prediction_counts['started'] += 1
    try:
        values = _boundary_original_infer_probs(model, xwins, device)
    except Exception:
        with _boundary_lock:
            _boundary_prediction_counts['failed'] += 1
        raise
    with _boundary_lock:
        _boundary_prediction_counts['completed'] += 1
    return values

def infer_pair_probs(model, plain, mirrored, device):
    assert plain.shape == mirrored.shape and 1 <= len(plain) <= 94
    assert plain.shape[1:] == (3, INPUT_RES, INPUT_RES)
    assert all(not m.training for m in model.modules())
    return 0.5 * (infer_probs(model, plain, device) + infer_probs(model, mirrored, device))

_boundary_qualifying = [True]
def _qualify_boundary_inputs():
    assert torch.cuda.device_count() == 2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert SLOT_SIZES == [26,22,18,12,18] and K_EVAL == 94 and INPUT_RES == 320
    centers = np.arange(1,95)
    legacy = series_donor_indices(centers,SLOT_SIZES)
    repaired = series_donor_indices(centers,SLOT_SIZES,mode='edge-repeat')
    assert np.array_equal(legacy,np.stack([centers-1,centers,centers+1],axis=1))
    assert same_series_windows(centers,SLOT_SIZES).sum() == 86 and (legacy != repaired).sum() == 8
    assert np.array_equal(repaired[:,1],centers)
    # Synthetic uint8 pixels only. CPU and GPU execute the actual modified
    # crop/LUT/mirror functions, including the all-crossing repeated-center edge.
    vol = np.random.default_rng(20261016).integers(0,256,size=(96,IMG,IMG),dtype=np.uint8)
    masks = [np.ones(96,np.uint8),np.eye(1,96,25,dtype=np.uint8).reshape(-1)]
    devices = []
    for index in range(2):
        dev = 'cuda:'+str(index)
        for mask in masks:
            for mirror in (False,True):
                cpu = eval_windows(vol,mask,94,INPUT_RES,norm=NORM,mirror=mirror)
                gpu = gpu_windows(vol,mask,94,INPUT_RES,dev,norm=NORM,mirror=mirror)
                assert torch.equal(cpu,gpu.cpu()), 'Actual CPU/GPU index/crop/LUT/mirror values differ'
                del cpu,gpu
        devices.append({'index':index,'name':torch.cuda.get_device_name(index),'actual_CPU_GPU_pixels_bit_equal':True})
        torch.cuda.empty_cache()
    _boundary_qualifying[0] = False
    return devices

_boundary_started = time.monotonic()
_boundary_devices = _qualify_boundary_inputs()
main()

# Keep original complete test.csv identity and probability contract; no labels.
import pandas as _boundary_pd
_boundary_root = find_test_root()
_boundary_ids = _boundary_pd.read_csv(_boundary_root+'/test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist()
_boundary_path = '/kaggle/working/submission.csv'
_boundary_output = _boundary_pd.read_csv(_boundary_path,dtype={'StudyInstanceUID':str})
assert _boundary_output.StudyInstanceUID.tolist() == _boundary_ids and len(set(_boundary_ids)) == len(_boundary_ids)
assert list(_boundary_output.columns) == ['StudyInstanceUID',*LAB]
assert np.isfinite(_boundary_output[LAB].values).all() and ((_boundary_output[LAB].values >= 0)&(_boundary_output[LAB].values <= 1)).all()
if len(_boundary_ids) == 3:
    assert _boundary_prediction_counts == {'started':6,'completed':6,'failed':0}, 'Public MRI model fallback is not acceptance'
receipt = {'status':'released_model_series_boundary_MRI_CSV_complete','policy':_BOUNDARY_POLICY,'parent_source_sha256':_BOUNDARY_PARENT_SHA,'helper_sha256':_BOUNDARY_HELPER_SHA,'checkpoint_sha256':_BOUNDARY_WEIGHTS_SHA,'own_training':False,'independent_internal_validation':False,'source_models_unchanged':True,'studies':len(_boundary_ids),'columns':13,'CSV_sha256':hashlib.sha256(open(_boundary_path,'rb').read()).hexdigest(),'CSV_bytes':os.stat(_boundary_path).st_size,'test_roster_sha256':hashlib.sha256(json.dumps(_boundary_ids,separators=(',',':')).encode()).hexdigest(),'prediction_counts':_boundary_prediction_counts,'devices':_boundary_devices,'window_records':_boundary_records,'elapsed_seconds':time.monotonic()-_boundary_started,'clinical_accuracy_claim':False,'competition_submission':False}
with open('/kaggle/working/series-boundary-accepted.json','x') as _f:json.dump(receipt,_f,indent=2)
print(json.dumps(receipt),flush=True)
"""
    replacements = {
        "BOUNDARY_POLICY_VALUE": policy,
        "PARENT_SHA_VALUE": PARENT_SOURCE_SHA256,
        "HELPER_SHA_VALUE": hashlib.sha256(helper.encode()).hexdigest(),
        "WEIGHTS_SHA_VALUE": WEIGHTS_SHA256,
    }
    for key, value in replacements.items():
        patch = patch.replace(key, repr(value))
    text = source + "\n\n" + helper_body + "\n" + patch
    compile(text, "series_boundary_reference", "exec")
    return text


def build(parent: Path, helper: Path, policy: str, destination: Path) -> dict:
    text = instrument(parent.read_text(), helper.read_text(), policy)
    if destination.exists():
        raise ValueError("Existing variant source; never overwrite a pushed version")
    destination.write_text(text)
    digest = hashlib.sha256(text.encode()).hexdigest()
    return {
        "policy": policy,
        "source_sha256": digest,
        "source_bytes": len(text.encode()),
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "checkpoint_sha256": WEIGHTS_SHA256,
        "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "kernel_slug": "godlydonuts/rsna-series-" + policy + "-" + digest[:12],
        "weights_changed": False,
        "own_training": False,
        "independent_internal_validation": False,
    }
