"""Private five-minute Kaggle GPU environment probe; no competition data."""

import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

import torch

result = {
    "schema_version": 1,
    "purpose": "runtime_environment_only_not_competition_submission",
    "python": platform.python_version(),
    "machine": platform.machine(),
    "torch": str(torch.__version__),
    "torch_cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "compiled_cuda_arches": torch.cuda.get_arch_list(),
    "gpus": [
        {
            "name": torch.cuda.get_device_name(i),
            "capability": list(torch.cuda.get_device_capability(i)),
            "total_memory": torch.cuda.get_device_properties(i).total_memory,
        }
        for i in range(torch.cuda.device_count())
    ],
    "packages": {},
    "competition_data_accessed": False,
    "competition_submission_created": False,
}
for package in ("numpy", "pandas", "transformers", "safetensors", "pydicom"):
    try:
        result["packages"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        result["packages"][package] = None
driver = subprocess.run(
    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
    capture_output=True,
    text=True,
    timeout=20,
    check=True,
)
result["driver_versions"] = sorted(set(driver.stdout.split()))
Path("/kaggle/working/environment.json").write_text(json.dumps(result, indent=2, sort_keys=True))
print(json.dumps(result, sort_keys=True))
