"""Compile a private original-pixel field qualification; never a competition entry."""

from __future__ import annotations

import ast
import hashlib

from scripts.kaggle.series_boundary_notebook import GUARD, PARENT_SOURCE_SHA256

CODEC_SHA = "7bdaf41a864b499401ff4a62e6288d30eead3849c0fda5c49cc67b3b2d0e2ab5"
RUNNER_SHA = "0354151fb11a68d42a90c46be641176e18a52337893c07ccd12ab20778e4871f"


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def build(parent, codec, runner):
    if digest(parent) != PARENT_SOURCE_SHA256:
        raise ValueError("Exact licensed0.949 MRI parent required")
    return _build(parent, codec, runner)


def _build(parent, codec, runner):
    if (
        digest(codec) != CODEC_SHA
        or digest(runner) != RUNNER_SHA
        or parent.count(GUARD) != 1
        or parent.count("# %% [code]") < 1
        or parent[parent.index(GUARD) + len(GUARD) :].strip()
    ):
        raise ValueError("Exact admitted codec, qualifier and single inert parent guard required")
    body = parent[parent.index("# %% [code]") : parent.index(GUARD)]
    assignments = [
        n
        for n in ast.parse(body).body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "INPUT_RES" for t in n.targets)
    ]
    if len(assignments) != 1 or ast.literal_eval(assignments[0].value) != 320:
        raise ValueError("One original320 field declaration required")
    source = (
        "# Private unchanged-weight original-MRI field qualification. No competition entry.\n"
        "import os,json\nfrom pathlib import Path\n"
        "os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\n"
        "_FOV_PARENT_BODY=" + repr(body) + "\n"
        "_FOV_CODEC=" + repr(codec) + "\n"
        "_FOV_RUNNER=" + repr(runner) + "\n"
        "if __name__=='__main__':\n"
        "    namespace={'__name__':'private_licensed_field_qualification'}\n"
        "    exec(compile(_FOV_RUNNER,'licensed_field_qualification','exec'),namespace)\n"
        "    result=namespace['qualify'](_FOV_PARENT_BODY,_FOV_CODEC)\n"
        "    Path('/kaggle/working/licensed-field-precision-accepted.json').write_text(json.dumps(result,sort_keys=True,indent=2))\n"
        "    print(json.dumps(result),flush=True)\n"
    )
    compile(source, "private_original_field_qualification", "exec")
    return source, {
        "source_sha256": digest(source),
        "source_bytes": len(source.encode()),
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "codec_sha256": CODEC_SHA,
        "runner_sha256": RUNNER_SHA,
        "roles": [320, 384],
        "original_MRI_studies": 32,
        "GPU_count": 2,
        "maximum_runtime_seconds": 1800,
        "competition_entry": False,
        "new_weight_or_image_Dataset_required": False,
        "clinical_accuracy_claim": False,
    }
