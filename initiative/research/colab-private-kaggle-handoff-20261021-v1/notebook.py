"""Build the credential-free Colab cell after actual canonical storage acceptance."""

import argparse
import base64
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build(pin, lease, receipt):
    spec = importlib.util.spec_from_file_location("Colab_cell_lease", HERE / "lease.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.payload(pin)
    assert lease["storage_control_git"] == pin
    assert lease["code_sha256"] == {
        n: hashlib.sha256(v.encode()).hexdigest() for n, v in files.items()
    }
    assert lease["input_manifest_sha256"] == receipt["files"]["input-manifest.bin"]["sha256"]
    assert receipt["status"] == "private_Kaggle_input_creation_returned_download_pending"
    package = {"files": files, "lease": lease, "receipt": receipt}
    body = json.dumps(package, sort_keys=True).encode()
    encoded = base64.b64encode(gzip.compress(body, mtime=0)).decode()
    return f"""# Independent full-data C100: seed20261021, 3887 studies, six epochs.
# Private inputs and checkpoints; no competition entry or billing change.
import os, sys, json, gzip, base64, hashlib, subprocess, importlib.util, getpass
from pathlib import Path
os.umask(0o077)
os.environ.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                  HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
control = Path('/content/rsna-colab-private-control-20261021-v1')
assert not control.exists(), 'Reconcile original execution before any continuation'
body = gzip.decompress(base64.b64decode({encoded!r}))
assert hashlib.sha256(body).hexdigest() == {hashlib.sha256(body).hexdigest()!r}
package = json.loads(body)
control.mkdir(mode=0o700)
for name, text in package['files'].items():
    assert '/' not in name and hashlib.sha256(text.encode()).hexdigest() == package['lease']['code_sha256'][name]
    with (control/name).open('x') as f: f.write(text)
with (control/'checkpoint-lease.private.json').open('x') as f: json.dump(package['lease'], f)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'kaggle==2.2.3',
                'kagglesdk==0.1.30', 'timm==1.0.29'], check=True)
import torch, timm
assert torch.__version__.startswith('2.11.0') and timm.__version__ == '1.0.29'
assert torch.cuda.device_count() == 1 and 'A100' in torch.cuda.get_device_name(0)
assert torch.cuda.get_device_properties(0).total_memory >= 75*2**30
print(json.dumps({{'phase':'actual_A100_runtime', 'torch':torch.__version__,
                  'timm':timm.__version__, 'GPU':torch.cuda.get_device_name(0),
                  'storage_control_git':package['lease']['storage_control_git']}}), flush=True)
sys.path.insert(0, str(control))
from colab_storage import download_inputs
HF_token = getpass.getpass('Private HF read token (RAM only): ')
Kaggle_token = getpass.getpass('Private Kaggle token (RAM only): ')
try:
    download_inputs(HF_token, Kaggle_token, package['receipt'])
    HF_token = None
    import actual_colab_train
    actual_colab_train.main(Kaggle_token)
except Exception as error:
    # Keep detailed runtime data private; do not print signed URLs or credentials.
    print(json.dumps({{'phase':'stopped_reconcile_original', 'type':type(error).__name__,
                      'automatic_retry':False}}), flush=True)
    raise RuntimeError('Original execution stopped; inspect private state before continuing') from None
finally:
    HF_token = None
    Kaggle_token = None
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cell = build(
        args.source_git, json.loads(args.lease.read_text()), json.loads(args.receipt.read_text())
    )
    with args.output.open("x") as f:
        f.write(cell)
