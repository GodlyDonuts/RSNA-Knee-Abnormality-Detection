"""Retain exact original modules and add isolated selected-frame assembly."""

import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = "e5385a8dc72e657059e4964bf0d754826b9cb5579fb4fd5a6a2e96ef91344de8"


def build(original_CPU_script):
    assert hashlib.sha256(original_CPU_script.encode()).hexdigest() == PARENT
    values = {}
    for node in ast.parse(original_CPU_script).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in ("FILES", "BINDING")
        ):
            assert node.targets[0].id not in values
            values[node.targets[0].id] = ast.literal_eval(node.value)
    files, binding = values["FILES"], values["BINDING"]
    assert binding["original_C_kernel"] == 140024153 and binding["original_C_version"] == 1
    assert set(files) == set(binding["module_sha256"])
    assert all(
        hashlib.sha256(s.encode()).hexdigest() == binding["module_sha256"][n]
        for n, s in files.items()
    )
    files["benchmark.py"] = (HERE / "benchmark.py").read_text()
    files["resilient_decoder.py"] = (
        HERE.parent / "predictive-inference-resilience-20261021-v1/decoder.py"
    ).read_text()
    binding = {
        **binding,
        "module_sha256": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in files.items()},
        "parent_descriptor_batch_CPU_source_sha256": PARENT,
        "scientific_model_changed": False,
        "bitwise_equivalence_required": True,
        "large_study_engineering_only": True,
    }
    script = (
        "import hashlib,json,os,sys\nfrom pathlib import Path\n"
        + f"FILES={files!r}\nBINDING={binding!r}\n"
        "def bootstrap():\n"
        " os.umask(0o077)\n"
        " os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')\n"
        " root=Path('/kaggle/working/predictive-large-study-CPU-v1')\n"
        " assert not root.exists()\n root.mkdir(mode=0o700)\n"
        " for n,body in FILES.items():\n"
        "  assert hashlib.sha256(body.encode()).hexdigest()==BINDING['module_sha256'][n]\n"
        "  p=root/n\n  p.parent.mkdir(parents=True,exist_ok=True)\n  p.write_text(body)\n"
        " sys.path.insert(0,str(root))\n import benchmark\n benchmark.main(BINDING)\n"
        "if __name__=='__main__': bootstrap()\n"
    )
    ast.parse(script)
    return script, {
        "status": "bounded_large_study_preparation_compiled",
        "binding": binding,
        "source_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "source_bytes": len(script.encode()),
        "GPU": False,
        "competition_entry": False,
        "new_Dataset": False,
    }
