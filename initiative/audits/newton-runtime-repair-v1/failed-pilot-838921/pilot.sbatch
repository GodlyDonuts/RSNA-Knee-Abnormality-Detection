#!/bin/bash
#SBATCH --job-name=rsna-dinov3-pilot-v4
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclude=evc38,evc50
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Run through Slurm}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose the assigned GPUs}"
cd -- /home/sa305415/rsna-knee/code
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
probe_allocated_driver() {
  "$@" /usr/bin/timeout --kill-after=5s 25s /usr/bin/python3.12 - <<'RSNA_ALLOCATED_DRIVER_PROBE'
"""Fast read-only CUDA/NVML visibility check inside the scheduler allocation."""
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

record = {
    'scope': 'allocated devices only; no visibility changes or GPU computation',
    'job_id': os.environ.get('SLURM_JOB_ID'),
    'node': os.uname().nodename,
    'allocation': {key: os.environ.get(key) for key in (
        'CUDA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS', 'SLURM_GPUS_ON_NODE',
        'SLURM_CPUS_PER_TASK', 'SLURM_STEP_GPUS', 'CUDA_DEVICE_ORDER')},
    'cpu_affinity': sorted(os.sched_getaffinity(0)),
}
errors = []
if not record['job_id'] or len((os.environ.get('CUDA_VISIBLE_DEVICES') or '').split(',')) != 2:
    errors.append('two scheduler-visible devices required')
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,pci.bus_id', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=15,
    )
    record['nvidia_inventory'] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    count = len([line for line in result.stdout.splitlines() if line.strip()])
    if result.returncode != 0 or count != 2:
        errors.append('NVML must expose both allocated GPUs')
except (OSError, subprocess.TimeoutExpired) as error:
    errors.append('NVML inventory failed: '+type(error).__name__)
try:
    driver = ctypes.CDLL('libcuda.so.1')
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    result = driver.cuInit(0)
    record['cuda_driver_init_code'] = result
    if result != 0:
        errors.append('CUDA driver initialization failed')
    else:
        count = ctypes.c_int()
        code = driver.cuDeviceGetCount(ctypes.byref(count))
        record['cuda_driver_count_code'] = code
        record['cuda_driver_device_count'] = count.value
        if code != 0 or count.value != 2:
            errors.append('CUDA driver must expose both allocated GPUs')
except (OSError, AttributeError) as error:
    errors.append('CUDA driver load failed: '+type(error).__name__)
try:
    record['mapped_libcuda_paths'] = sorted({
        line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
        if '/libcuda.so' in line and line.split()[-1].startswith('/')
    })
except OSError:
    record['mapped_libcuda_paths'] = None
if errors:
    record['device_nodes'] = []
    for path in sorted(Path('/dev').glob('nvidia[0-9]*')):
        try:
            st = path.stat()
            record['device_nodes'].append({'path': str(path), 'major': os.major(st.st_rdev),
                                          'minor': os.minor(st.st_rdev), 'mode': oct(st.st_mode & 0o777)})
        except OSError:
            pass
    try:
        record['self_cgroup'] = Path('/proc/self/cgroup').read_text()
        record['readable_devices_lists'] = []
        for line in record['self_cgroup'].splitlines():
            _, controllers, relative = line.split(':', 2)
            if 'devices' in controllers.split(',') and '..' not in Path(relative).parts:
                path = Path('/sys/fs/cgroup/devices') / relative.lstrip('/') / 'devices.list'
                if path.is_file():
                    record['readable_devices_lists'].append({'path': str(path), 'contents': path.read_text()})
    except OSError:
        record['self_cgroup'] = None
    record['readable_driver_gpu_metadata'] = []
    for path in sorted(Path('/proc/driver/nvidia/gpus').glob('*/information')):
        try:
            rows = [line for line in path.read_text().splitlines()
                    if line.startswith(('Model:', 'GPU UUID:', 'Bus Location:', 'Device Minor:'))]
            record['readable_driver_gpu_metadata'].append({'path': str(path), 'fields': rows})
        except OSError:
            pass
record['errors'] = errors
record['status'] = 'passed' if not errors else 'blocked'
print(json.dumps(record, sort_keys=True), flush=True)
sys.exit(2 if errors else 0)
RSNA_ALLOCATED_DRIVER_PROBE
}
if ! probe_allocated_driver; then
  probe_allocated_driver srun --nodes=1 --ntasks=1 --cpus-per-task=4 --gres=gpu:nvidia_h100_pcie:2 || true
  exit 2
fi
supervisors=()
stop_requested=0
request_checkpoint() {
  stop_requested=1
  for pid in "${supervisors[@]}"; do kill -USR1 "$pid" 2>/dev/null || true; done
}
trap request_checkpoint USR1 TERM INT
runtime_root=
environment_root=
cleanup_runtime() {
  local previous_status="$?"
  if [[ -n "$runtime_root" ]]; then
    /usr/bin/python3.12 scripts/newton/runtime_archive.py cleanup --root "$runtime_root" || previous_status=2
  fi
  if [[ -n "$environment_root" ]]; then
    /usr/bin/python3.12 scripts/newton/runtime_environment.py cleanup --root "$environment_root" || previous_status=2
  fi
  return "$previous_status"
}
trap cleanup_runtime EXIT
environment_json=$(/usr/bin/python3.12 scripts/newton/runtime_environment.py extract --archive /home/sa305415/rsna-knee/kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar --sha256 89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338 --expected-files 40403 --expected-bytes 6322691335 --scratch-base /tmp --shared-python /usr/bin/python3.12 --min-free-gib 20)
environment_root=$(/usr/bin/python3.12 -c 'import json,sys; print(json.load(sys.stdin)["runtime_root"])' <<< "$environment_json")
runtime_python="$environment_root/environment/bin/python"
if (( stop_requested )); then exit 75; fi
runtime_json=$(/usr/bin/python3.12 scripts/newton/runtime_archive.py extract --archive /home/sa305415/rsna-knee/cache/pilot100-fold0-v1.tar --sha256 8d50e027f66e91beb6afa2b56877b8da59059f6d43e7c13aa0103059d447d715 --expected-files 2248 --expected-bytes 4077105588 --scratch-base /tmp --min-free-gib 20)
runtime_root=$(/usr/bin/python3.12 -c 'import json,sys; print(json.load(sys.stdin)["runtime_root"])' <<< "$runtime_json")
runtime_cache="$runtime_root/cache"
"$runtime_python" scripts/newton/preflight.py --device cuda --expected-gpus 2 --repo /home/sa305415/rsna-knee/code --expected-git e7b08625854de692b9266ccfc25c90fc22169f58 --cache "$runtime_cache" --cache-manifest /home/sa305415/rsna-knee/code/initiative/audits/full-image-recovery-v1/audit/series_cache_index.parquet --storage /home/sa305415/rsna-knee/logs --min-free-gb 20 --config /home/sa305415/rsna-knee/encoder-stage/run-configs/pilot100-fold0.json --require-nccl
if (( stop_requested )); then exit 75; fi
IFS="," read -r -a assigned_gpus <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#assigned_gpus[@]} != 2 )); then exit 2; fi
"$runtime_python" scripts/local_competitive_acceptance.py --cache "$runtime_cache" --output /home/sa305415/rsna-knee/runs/dinov3-l16-pilot100-fold0-e7b08625-v4 --world-size 2 --config /home/sa305415/rsna-knee/encoder-stage/run-configs/pilot100-fold0.json --device cuda --skip-public-examples &
supervisors+=("$!")
if (( stop_requested )); then kill -USR1 "$!" 2>/dev/null || true; fi
result=0
for pid in "${supervisors[@]}"; do
  code=0
  while true; do
    set +e
    wait "$pid"
    code=$?
    set -e
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
  done
  if (( code != 0 && code != 75 )); then result=$code;
  elif (( code == 75 && result == 0 )); then result=75; fi
done
exit "$result"
