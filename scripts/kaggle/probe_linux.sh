#!/bin/bash -l
# Read-only machine probe inside a zero-GPU, bounded Slurm allocation.
set -euo pipefail
umask 077
module load python/python-3.11.4-gcc-12.2.0
python3 - <<'PY'
import json, os, platform, shutil
from pathlib import Path
driver = Path('/proc/driver/nvidia/version')
print(json.dumps({
    'schema_version': 1,
    'stage': 'linux_wheelhouse_environment_probe',
    'job_id': os.environ.get('SLURM_JOB_ID'),
    'allocated_cpus': os.environ.get('SLURM_CPUS_PER_TASK'),
    'allocated_gpus': os.environ.get('SLURM_JOB_GPUS', ''),
    'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
    'python': platform.python_version(),
    'platform': platform.platform(),
    'architecture': platform.machine(),
    'driver_module': driver.read_text().strip() if driver.exists() else None,
    'tmp_free_bytes': shutil.disk_usage('/tmp').free,
    'python312_present': shutil.which('python3.12') is not None,
    'uv_present': shutil.which('uv') is not None,
}, sort_keys=True))
PY
