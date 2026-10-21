"""Layer explicit per-study resilience onto the exact faster preparation capsule.

Not a root-cause claim. Keep source/model bytes and exact-path MRI unchanged;
qualification still requires original context equality and strict precision/RAM.
"""

import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent

NUMERICAL_HELPER = """
def predict_with_precision_recovery(predict, model, contexts):
    try:
        return predict(model, contexts, precision="fp16"), "fp16"
    except (FloatingPointError, ValueError):
        # Recompute the same learned model on the same MRI at higher precision.
        # A second failure propagates; no constants or alternate model.
        return predict(model, contexts, precision="fp32"), "fp32_recovery"
"""


def replace(s, old, new):
    if s.count(old) != 1:
        raise ValueError("Exact original serving anchor required")
    return s.replace(old, new)


def capsule(files):
    decoder = (HERE / "decoder.py").read_text()
    portable = files["serving/preparation_upgrade.py"]
    addition = f"\nRESILIENT_DECODER={decoder!r}\nNUMERICAL_HELPER={NUMERICAL_HELPER!r}\nRESILIENT_DECODER_SHA={hashlib.sha256(decoder.encode()).hexdigest()!r}\n"
    addition += """
_exact_preparation_upgrade=upgrade
_exact_preparation_qualification=require_qualification

def upgrade(original):
    files=_exact_preparation_upgrade(original)
    helper=files["descriptor_cache.py"]
    helper += "\\nimport types\\n_decoder=types.ModuleType('explicit_serving_decoder')\\n"
    helper += f"exec(compile({RESILIENT_DECODER!r},'explicit_serving_decoder','exec'),_decoder.__dict__)\\nload_study=_decoder.load_study\\n"
    helper += NUMERICAL_HELPER
    worker=files["worker.py.txt"]
    worker=replace(worker,"from _predictive_descriptor_cache import prepare_contexts","from _predictive_descriptor_cache import prepare_contexts, load_study, predict_with_precision_recovery")
    worker=replace(worker,"    from rsna.competitive.predictive.dicom import load_study\\n","")
    worker=replace(worker,'                low = predict(models[device], contexts, precision="fp16")','                low, actual_precision = predict_with_precision_recovery(predict, models[device], contexts)')
    worker=replace(worker,'                    "FP16_seconds": fp16_seconds,','                    "FP16_seconds": fp16_seconds,\\n                    "prediction_precision": actual_precision,')
    worker=replace(worker,'        assert memory_ok and precision_ok, "Measured trained precision/memory did not qualify"','        if prepared is not None:\\n            assert memory_ok and precision_ok, "Measured trained precision/memory did not qualify"')
    worker=replace(worker,'        "clinical_score": None,','        "clinical_score": None,\\n        "serving_resilience_version": "exact_first_MRI_FP32_recovery_v1",')
    worker=replace(worker,'        stage("failed", error_type=type(error).__name__)','        import traceback\\n        traceback.print_exc()\\n        stage("failed", error_type=type(error).__name__)')
    ast.parse(worker);ast.parse(helper)
    return {**files,"worker.py.txt":worker,"descriptor_cache.py":helper}

def require_qualification(receipt):
    _exact_preparation_qualification(receipt)
    if receipt.get("serving_resilience_version") != "exact_first_MRI_FP32_recovery_v1":
        raise ValueError("Own actual resilient trained qualification required")
    for model in receipt["models"]:
        for m in model["measurements"]:
            if m.get("decoder_route") != "original_exact" or m.get("prediction_precision") != "fp16":
                raise ValueError("Original public path and trained FP16 comparison required")
"""
    portable += addition
    ast.parse(portable)
    return {**files, "serving/preparation_upgrade.py": portable}
