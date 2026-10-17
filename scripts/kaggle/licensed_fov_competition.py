"""Serve the fixed full384 field only after actual unchanged-weight T4 acceptance."""

from __future__ import annotations

import hashlib

from scripts.kaggle.licensed_fov_notebook import RUNNER_SHA, digest
from scripts.kaggle.released_context_notebook import bodies
from scripts.kaggle.series_boundary_notebook import GUARD, PARENT_SOURCE_SHA256, WEIGHTS_SHA256

QUALIFIED_SOURCE = "47e79180cca2808f25f45c8b11ca5804750eca206c6333c59b9c9cface95859b"
BOUNDED_SHA = "5576dff1348ebb7fa322c0ce78f7fdf8aa4b9ef1f2a6be2b757765fcc733cd42"


def accepted_parameter_hash(actual):
    if actual.get("status") != "COMPLETE" or actual.get("source_sha256") != QUALIFIED_SOURCE:
        raise ValueError("Exact actual original field qualification must have completed")
    receipt = actual.get("receipt", {})
    devices = receipt.get("devices", [])
    if (
        receipt.get("status")
        != "actual_original32_bothT4_unchanged_weights_full384_field_qualified"
        or receipt.get("parent_checkpoint_sha256") != WEIGHTS_SHA256
        or receipt.get("loaded_parameter_count") != 73136096
        or receipt.get("studies") != 32
        or receipt.get("maximum_cross_GPU384_probability_difference", 1) > 1e-5
        or receipt.get("competition_entry") is not False
        or len(devices) != 2
    ):
        raise ValueError("Actual complete unchanged-parent both-T4 qualification required")
    for index, device in enumerate(devices):
        roles = device.get("roles", [])
        if (
            device.get("index") != index
            or "T4" not in device.get("GPU", "")
            or device.get("all_learned_parameters_equal_320_384") is not True
            or device.get("control_probability_error", 1) > 0.003
            or [role.get("input_side") for role in roles] != [320, 384]
            or not all(
                role.get("CPU_GPU_inputs_bit_equal") is True
                and role.get("original320_input_hashes_reproduced") is True
                and role.get("studies") == 32
                and role.get("peak_GPU_bytes", 16 * 2**30) < 14 * 2**30
                for role in roles
            )
        ):
            raise ValueError(
                "Both original controls, unchanged tensors and safe384 inputs required"
            )
    parameter_hash = devices[0].get("learned_parameter_sha256", "")
    if (
        len(parameter_hash) != 64
        or any(c not in "0123456789abcdef" for c in parameter_hash)
        or devices[1].get("learned_parameter_sha256") != parameter_hash
    ):
        raise ValueError("Both devices must bind the same complete learned tensors")
    return parameter_hash


def build(parent, bounded, runner, actual):
    if (
        digest(parent) != PARENT_SOURCE_SHA256
        or digest(bounded) != BOUNDED_SHA
        or digest(runner) != RUNNER_SHA
        or parent.count(GUARD) != 1
        or parent[parent.index(GUARD) + len(GUARD) :].strip()
    ):
        raise ValueError("Exact licensed MRI graph and admitted private field qualifier required")
    return _build(parent, bounded, runner, actual)


def _build(parent, bounded, runner, actual):
    parameter_hash = accepted_parameter_hash(actual)
    source = parent[parent.index("# %% [code]") : parent.index(GUARD)]
    replacements = {
        "INPUT_RES = 320": "INPUT_RES = 384",
        "assert plain.shape==mirrored.shape==(94,3,320,320)": "assert plain.shape==mirrored.shape==(94,3,INPUT_RES,INPUT_RES)",
        "nproc = max(4, min(96, (os.cpu_count() or 8) - 8))": "nproc = 2",
        "pool.imap_unordered(_pre_one, test_ids, chunksize=2)": "bounded_results(pool, _pre_one, test_ids, limit=4)",
        "q = queue.Queue(maxsize=QUEUE_MAX)": "q = queue.Queue(maxsize=2)",
        'print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)': 'print(json.dumps({"field_MRI_fallback_index":i,"error_type":type(e).__name__}),flush=True)',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("Unique original field, shape and bounded-memory sites required")
        source = source.replace(old, new)
    patch = r"""
_fov_parameter_hash=PARAMETER_VALUE
_fov_original_load=load_model
_fov_original_pair=infer_pair_probs
_fov_original_gpu_windows=gpu_windows
_fov_counts={'models_loaded':0,'pairs_started':0,'pairs_completed':0,'pair_failures':0,'empty_views':0}
_fov_lock=threading.Lock()

def load_model(*args,**kwargs):
    model,side=_fov_original_load(*args,**kwargs)
    assert side==384 and sum(p.numel() for p in model.parameters())==73136096
    assert parameter_digest(model)==_fov_parameter_hash and all(not m.training for m in model.modules())
    with _fov_lock:_fov_counts['models_loaded']+=1
    return model,side

def gpu_windows(vol,mask,k,res,dev,norm=NORM,mirror=False):
    assert k==94 and res==384 and vol.shape==(96,384,384)
    if not mask.any():
        with _fov_lock:_fov_counts['empty_views']+=1
    return _fov_original_gpu_windows(vol,mask,k,res,dev,norm=norm,mirror=mirror)

def infer_pair_probs(model,plain,mirrored,device):
    with _fov_lock:_fov_counts['pairs_started']+=1
    try:
        values=_fov_original_pair(model,plain,mirrored,device)
        assert values.shape==(12,) and np.isfinite(values).all() and ((values>=0)&(values<=1)).all()
    except Exception:
        with _fov_lock:_fov_counts['pair_failures']+=1
        raise
    with _fov_lock:_fov_counts['pairs_completed']+=1
    return values

def _pre_one(sid):
    try:vol,mask=_READER(sid,_SER,_TSDIR)
    except Exception as exc:
        print(json.dumps({'field_preprocessing_error_type':type(exc).__name__}),flush=True)
        vol=np.zeros((MAXS,IMG,IMG),np.uint8);mask=np.zeros(MAXS,np.uint8)
    return sid,vol,mask

def _fov_competition():
    import pandas as pd
    assert torch.cuda.device_count()==2 and all('T4' in torch.cuda.get_device_name(i) for i in range(2))
    assert INPUT_RES==384 and IMG==384 and K_EVAL==94 and SLOT_SIZES==[26,22,18,12,18] and len(ARMS)==1
    torch.set_num_threads(1);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    started=time.monotonic();main()
    root=find_test_root();ids=pd.read_csv(str(root)+'/test.csv',dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist();path=Path('/kaggle/working/submission.csv');out=pd.read_csv(path,dtype={'StudyInstanceUID':str})
    assert out.StudyInstanceUID.tolist()==ids and len(set(ids))==len(ids) and list(out.columns)==['StudyInstanceUID',*LAB]
    assert np.isfinite(out[LAB].values).all() and ((out[LAB].values>=0)&(out[LAB].values<=1)).all()
    if len(ids)==3:assert _fov_counts=={'models_loaded':2,'pairs_started':3,'pairs_completed':3,'pair_failures':0,'empty_views':0},'Public field fallbacks do not qualify'
    result={'status':'actual_unchanged_weight_full384_MRI_test_roster_CSV_complete','parent_source_sha256':PARENT_VALUE,'parent_checkpoint_sha256':WEIGHTS_VALUE,'qualification_source_sha256':QUALIFICATION_VALUE,'learned_parameter_sha256':_fov_parameter_hash,'input_side':384,'windows_per_view':94,'views':2,'studies':len(ids),'columns':13,'CSV_bytes':path.stat().st_size,'CSV_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'test_roster_sha256':hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),'counts':_fov_counts,'CPU_decode_errors_logged_by_type_not_aggregated_across_processes':True,'elapsed_seconds':time.monotonic()-started,'independent_internal_validation':False,'clinical_accuracy_claim':False,'competition_entry':False}
    Path('/kaggle/working/licensed-field-MRI-accepted.json').write_text(json.dumps(result,sort_keys=True,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':_fov_competition()
"""
    for key, value in {
        "PARAMETER_VALUE": parameter_hash,
        "PARENT_VALUE": PARENT_SOURCE_SHA256,
        "WEIGHTS_VALUE": WEIGHTS_SHA256,
        "QUALIFICATION_VALUE": QUALIFIED_SOURCE,
    }.items():
        patch = patch.replace(key, repr(value))
    text = (
        "# Unchanged licensed MRI parent, fixed full384 input field. Historical source prose below.\n"
        "import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\nfrom pathlib import Path\nimport hashlib\n"
        + source
        + "\n"
        + bodies(bounded)
        + "\n"
        + bodies(runner, selected={"parameter_digest"})
        + "\n"
        + patch
    )
    compile(text, "qualified_unchanged_weight_field_MRI", "exec")
    return text, {
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_bytes": len(text.encode()),
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "parent_checkpoint_sha256": WEIGHTS_SHA256,
        "qualification_source_sha256": QUALIFIED_SOURCE,
        "learned_parameter_sha256": parameter_hash,
        "input_side": 384,
        "original_94_windows_two_views_and_rank_output": True,
        "new_Dataset_upload_required": False,
        "competition_entry": False,
        "independent_internal_validation": False,
    }
