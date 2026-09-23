#!/bin/bash
#SBATCH --job-name=rsna-evc38-cuda-health
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=evc38
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Run through Slurm}"
: "${CUDA_VISIBLE_DEVICES:?Two assigned GPUs required}"
cd /home/sa305415/rsna-knee/code-stopfix-1f486765
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/usr/bin/python3.12 - <<'RSNA_DRIVER_CHECK'
import ctypes
import json
import os
import subprocess

visible = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").split(",")
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=index,name,uuid", "--format=csv,noheader"],
    capture_output=True,
    text=True,
    timeout=15,
)
driver = ctypes.CDLL("libcuda.so.1")
driver.cuInit.argtypes = [ctypes.c_uint]
driver.cuInit.restype = ctypes.c_int
driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
driver.cuDeviceGetCount.restype = ctypes.c_int
init = driver.cuInit(0)
count = ctypes.c_int(-1)
count_code = driver.cuDeviceGetCount(ctypes.byref(count)) if init == 0 else -1
record = {
    "status": "passed" if len(visible) == 2 and result.returncode == 0 and len(result.stdout.splitlines()) == 2 and init == 0 and count_code == 0 and count.value == 2 else "blocked",
    "job_id": os.environ["SLURM_JOB_ID"],
    "node": os.uname().nodename,
    "visible_devices": visible,
    "nvidia_smi_returncode": result.returncode,
    "nvidia_smi_count": len(result.stdout.splitlines()),
    "cuda_init_code": init,
    "cuda_count_code": count_code,
    "cuda_device_count": count.value,
}
print(json.dumps(record, sort_keys=True), flush=True)
if record["status"] != "passed":
    raise SystemExit(2)
RSNA_DRIVER_CHECK
/usr/bin/python3.12 - <<'RSNA_PROBE_HASH'
import hashlib
from pathlib import Path

path = Path('/home/sa305415/rsna-knee/production-launch-b740-fold1-v2/cuda_context_probe.py')
assert hashlib.sha256(path.read_bytes()).hexdigest() == '6cecdc2db621c14b2420b3f729156315a3cf37023fbf36f142cef009e1d2caf2'
RSNA_PROBE_HASH
runtime_root=
cleanup() {
  local prior="$?"
  if [[ -n "$runtime_root" ]]; then
    /usr/bin/python3.12 scripts/newton/runtime_environment.py cleanup --root "$runtime_root" || prior=2
  fi
  return "$prior"
}
trap cleanup EXIT
result=$(/usr/bin/python3.12 scripts/newton/runtime_environment.py extract --archive /home/sa305415/rsna-knee/kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar --sha256 89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338 --expected-files 40403 --expected-bytes 6322691335 --scratch-base /tmp --shared-python /usr/bin/python3.12 --min-free-gib 20)
runtime_root=$(/usr/bin/python3.12 -c 'import json,sys; print(json.load(sys.stdin)["runtime_root"])' <<< "$result")
"$runtime_root/environment/bin/python" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 /home/sa305415/rsna-knee/production-launch-b740-fold1-v2/cuda_context_probe.py
