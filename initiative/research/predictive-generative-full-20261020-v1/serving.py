"""Compile a preserved G100 final with the unchanged original MRI/T4 worker.

Only training provenance differs from C100: G100 routes gradients differently,
but has the same learned-state graph and forward calculation. Derive a separate
standalone contract and compiler from pinned sources; never patch live modules.
There are no account calls, allocations, model reads or file writes here.
"""

import ast
import hashlib
import subprocess
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PIN = "2662bf7776903163ac4673571553a277e05a1176"
PREFIX = "initiative/research/predictive-density100-serving-20261020-v1/"
PINS = {
    PREFIX + "contract.py": "881a88f511e5bbdaf134d955a75508ff0db9636eee00c66254f8a2d455fea0c9",
    PREFIX + "compile.py": "c9bd605173b5a27ca3e89fd05911ed9b33f8af8c623b81b75c6825d671eb1fe7",
    "initiative/research/predictive-generative-full-20261020-v1/contract.py":
        "282cc27c41f17079631b89c06c1447987e2a48c90b11eafffb8954114012385d",
}


def source(name):
    raw = subprocess.check_output(["git", "show", PIN + ":" + name], cwd=REPO)
    if hashlib.sha256(raw).hexdigest() != PINS[name]:
        raise ValueError("Pinned serving derivation source differs")
    return raw.decode()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Expected one original serving derivation anchor")
    return text.replace(old, new, 1)


def function(text, name):
    matches = [n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(matches) != 1:
        raise ValueError("Expected one original function")
    return ast.get_source_segment(text, matches[0])


def contract_source():
    text = source(PREFIX + "contract.py")
    full = source("initiative/research/predictive-generative-full-20261020-v1/contract.py")
    # Use the very same actual 600-update pilot validation as the full trainer.
    accepted = '''def accepted_pilot(text, expected_sha, scheduler):
    require(isinstance(text, str) and hexadecimal(expected_sha, 64),
            "Explicit original G100 pilot bytes and SHA required")
    require(digest(text) == expected_sha, "Actual G100 pilot bytes differ")
    pilot = json.loads(text)
    completed(scheduler, PILOT_JOB)
    validate_pilot(pilot, {**scheduler, "GPU_job": scheduler["job_id"]})
    return pilot'''
    text = replace_once(text, function(text, "accepted_pilot"), accepted)
    text = text.replace("C100", "G100").replace("density100_pilot", "generative_pilot")
    text = replace_once(text, "import hashlib\n", "import hashlib\nimport math\n")
    text = replace_once(text, 'PILOT_SOURCE = "068a14a75e4d2b3bd83dda7783528956f7a2ca0f"',
                        'PILOT_SOURCE = "473a9a94d90890e8a97e7ed46d157dcf417ad173"')
    text = replace_once(text, 'PILOT_JOB = "981620"', 'PILOT_JOB = "986011"')
    text = replace_once(text, 'PILOT_BINDING = "e11c7c7fe5c1fd7cae87df313fb5d42be7128b7906f8d716a486b0857bbdd37a"',
                        'PILOT_BINDING = "41b7f830956bc3c7b47f6b90453cfe7fe00a2399510c5939379872719a49e4c6"')
    routing = '''
ROUTING_SHA = "7a88c915209f2e51a293a046402f58d03a588268281d627ea3ac122d8c3add86"
ROUTING_PROVENANCE = {
    "routing_sha256": ROUTING_SHA,
    "exam_gradients_through_Bayes_factor": False,
    "proper_conditional_density_gradients_attached": True,
}
'''
    text = replace_once(text, "FINAL_MATCH_FIELDS = (", routing + "\nFINAL_MATCH_FIELDS = (")
    text = replace_once(text, '    "arm", "branch", "GPU_job", "source_git", "binding_sha256", "recipe",',
                        '    *ROUTING_PROVENANCE,\n'
                        '    "arm", "branch", "GPU_job", "source_git", "binding_sha256", "recipe",')
    text = replace_once(text, '            "arm": "C", "branch": "G100", "source_git": self.full_source_git,',
                        '            **ROUTING_PROVENANCE,\n'
                        '            "arm": "C", "branch": "G100", "source_git": self.full_source_git,')
    text = replace_once(text, '        chosen = self._base.candidate(publication, provenance_text, own_source_text, license_bytes)',
                        '        require(p["code_sha256"].get("rsna/competitive/predictive/generative_routing.py")\n'
                        '                == ROUTING_SHA, "Actual trained routing source differs")\n'
                        '        chosen = self._base.candidate(publication, provenance_text, own_source_text, license_bytes)')
    text += "\n\n" + function(full, "validate_pilot") + "\n"
    ast.parse(text)
    return text


def compiler():
    adapter_source = contract_source()
    adapter = types.ModuleType("isolated_G100_serving_contract")
    exec(compile(adapter_source, "isolated_G100_serving_contract", "exec"), adapter.__dict__)
    text = source(PREFIX + "compile.py")
    start = text.index("spec = importlib.util.spec_from_file_location(")
    stop = text.index("spec.loader.exec_module(C)", start) + len("spec.loader.exec_module(C)")
    text = text[:start] + "C = ADAPTER_MODULE" + text[stop:]
    text = text.replace("C100", "G100")
    text = replace_once(text, 'adapter_source = (HERE / "contract.py").read_text()',
                        "adapter_source = ADAPTER_SOURCE")
    module = types.ModuleType("isolated_G100_serving_compiler")
    module.__file__ = str(HERE / "serving.py")
    module.ADAPTER_MODULE = adapter
    module.ADAPTER_SOURCE = adapter_source
    exec(compile(text, "isolated_G100_serving_compiler", "exec"), module.__dict__)
    return module


def build(parent_source, export, **options):
    """CPU source compilation only; actual final/qualification evidence is required."""
    compiled = compiler()
    script, receipt = compiled.build(parent_source, export, **options)
    return script, {**receipt, "derivation_source_git": PIN, "derivation_sha256": dict(PINS)}
