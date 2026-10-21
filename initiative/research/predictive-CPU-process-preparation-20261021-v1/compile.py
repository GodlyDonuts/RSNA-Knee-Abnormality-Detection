"""Extract only exact own CPU input modules from the actually accepted C source."""

import ast
import hashlib
from pathlib import Path

SOURCE = "c414ecfe6d7fff62c1f3b68a19f4874a79a6bb28e00a32df1917c3da592d6509"
HERE = Path(__file__).resolve().parent


def build(kernel, qualification):
    meta, source = kernel["metadata"], kernel["blob"]["source"]
    assert meta["id"] == 140024153 and meta["currentVersionNumber"] == 1
    assert hashlib.sha256(source.encode()).hexdigest() == SOURCE
    values = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("FILES", "BINDING"):
                assert name not in values
                values[name] = ast.literal_eval(node.value)
    binding = values["BINDING"]
    assert binding["qualification"] == qualification
    assert binding["mode"] == "competition" and binding["models"][0]["arm"] == "C"
    own = values["FILES"]
    assert set(own) == set(binding["module_sha256"])
    assert all(
        hashlib.sha256(body.encode()).hexdigest() == binding["module_sha256"][n]
        for n, body in own.items()
    )
    # No parent body, license, model or raw MRI is embedded. Modules are not changed.
    assert all(n.startswith("rsna/") or n.startswith("_predictive_") for n in own)
    own["benchmark.py"] = (HERE / "benchmark.py").read_text()
    observed = {
        "original_C_kernel": 140024153,
        "original_C_version": 1,
        "original_C_source_sha256": SOURCE,
        "module_sha256": {n: hashlib.sha256(s.encode()).hexdigest() for n, s in own.items()},
        "public_roster_sha256": binding["public_roster_sha256"],
        "torch": qualification["torch"],
        "pydicom": qualification["pydicom"],
        "pinned_Docker": qualification["pinned_Docker"],
        "scientific_preparation_changed": False,
    }
    script = (
        "import hashlib,json,os,sys\nfrom pathlib import Path\n"
        f"FILES={own!r}\nBINDING={observed!r}\n"
        "def bootstrap():\n"
        " os.umask(0o077)\n"
        " os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')\n"
        " root=Path('/kaggle/working/predictive-CPU-process-v1')\n"
        " assert not root.exists()\n root.mkdir(mode=0o700)\n"
        " for n,body in FILES.items():\n"
        "  assert hashlib.sha256(body.encode()).hexdigest()==BINDING['module_sha256'][n]\n"
        "  p=root/n\n  p.parent.mkdir(parents=True,exist_ok=True)\n  p.write_text(body)\n"
        " sys.path.insert(0,str(root))\n import benchmark\n benchmark.main(BINDING)\n"
        "if __name__=='__main__': bootstrap()\n"
    )
    ast.parse(script)
    return script, {
        "status": "original_input_CPU_comparison_compiled",
        "binding": observed,
        "source_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "source_bytes": len(script.encode()),
        "GPU": False,
        "competition_entry": False,
        "new_Dataset": False,
    }
