#!/bin/bash
#SBATCH --job-name=rsna-a32-all-pilot-v2
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=evc31
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --signal=B:USR1@300
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
: "${SLURM_JOB_ID:?Run through Slurm}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose the assigned GPUs}"
cd -- /home/sa305415/rsna-knee/code-adapter-a32-all-8911f90
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export GIT_CONFIG_COUNT=3 GIT_CONFIG_KEY_0=core.preloadIndex GIT_CONFIG_VALUE_0=false GIT_CONFIG_KEY_1=index.threads GIT_CONFIG_VALUE_1=1 GIT_CONFIG_KEY_2=status.renames GIT_CONFIG_VALUE_2=false
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
if not record['job_id'] or len((os.environ.get('CUDA_VISIBLE_DEVICES') or '').split(',')) != 1:
    errors.append('one scheduler-visible device required')
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,pci.bus_id', '--format=csv,noheader'],
        capture_output=True, text=True, timeout=15,
    )
    record['nvidia_inventory'] = {'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}
    count = len([line for line in result.stdout.splitlines() if line.strip()])
    if result.returncode != 0 or count != 1:
        errors.append('NVML must expose the allocated GPU')
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
        if code != 0 or count.value != 1:
            errors.append('CUDA driver must expose the allocated GPU')
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
  probe_allocated_driver srun --nodes=1 --ntasks=1 --cpus-per-task=2 --gres=gpu:nvidia_h100_pcie:1 || true
  exit 2
fi
CPATH=$(/usr/bin/python3.12 - <<'RSNA_HEADER_GUARD'
import hashlib,json,os,platform,struct,subprocess,sys,sysconfig
from pathlib import Path
root=Path('/home/sa305415/rsna-knee/python-headers-3.12.13-2-v3').resolve();pins={'cpu_proof_sha256': '95c1d9aa2a9676cbdc4fa46e3d3ffda0bb721f24221365953ce23819c01f1695', 'header_tree_sha256': '08c760c1938c53debf9a2ed133609d7ed07049902b4eae54da6ee93454372241', 'staging_receipt_sha256': '85d88b7d4076564f86ddc49e2e670a855617d752f38766856276de03f10768a3', 'generated_header_provenance_sha256': '38913bdb4e683d0c00446de75e0b103469e81a7eb5f47c4fc5c5828e25af47bb', 'rpm_sha256': 'd822aa263492b730c391f19d7614ad05c9acdc33ba165964565548ffa54ea3e3', 'generated_header_sha256': 'bf85b811c1ac1c8c5f62935da14ee3738305bb2ffd9bfb63af6ebad2a475f681', 'cpu_proof_job': 839074, 'remote_root': '/home/sa305415/rsna-knee/python-headers-3.12.13-2-v3'}
control=Path('/home/sa305415/rsna-knee/production-launch-adapter-a32-all-8911f90-v1')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,v):
 with (control/n).open('x') as out:json.dump(v,out,indent=2,sort_keys=True);out.write('\n');out.flush();os.fsync(out.fileno())
assert os.environ.get('SLURM_JOB_ID')
assert sha(root/'cpu-proof.json')==pins['cpu_proof_sha256']
assert sha(root/'staging-receipt.json')==pins['staging_receipt_sha256']
assert sha(root/'system-header-verified.json')==pins['generated_header_provenance_sha256']
proof=json.loads((root/'cpu-proof.json').read_text());source=json.loads((root/'system-header-verified.json').read_text())
assert proof['status']=='actual_triton_helper_compile_and_import_passed' and proof['actual_installed_builder_replayed'] and proof['requested_gpus']==0
assert proof['header_tree_sha256']==pins['header_tree_sha256'] and proof['rpm_sha256']==pins['rpm_sha256']
assert not proof['system_mutations'] and not proof['backend_disable_or_replacement']
assert sha(root.parent/'python-headers-3.12.13-2-v2/python3.12-devel-3.12.13-2.el8_10.x86_64.rpm')==pins['rpm_sha256']
manifest=json.loads((root/'header-tree.json').read_text());rows=manifest['files']
assert len(rows)==215 and hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()==manifest['canonical_tree_sha256']==pins['header_tree_sha256']
actual=[];extract=root/'extracted'
for p in sorted(extract.rglob('*')):
 assert not p.is_symlink()
 if p.is_file():actual.append({'path':str(p.relative_to(extract)),'bytes':p.stat().st_size,'sha256':sha(p)})
assert actual==rows
headers=extract/'usr/include/python3.12'
assert sha(headers/'pyconfig-64.h')==pins['generated_header_sha256']
mapped=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/libpython3.12.so' in line and line.split()[-1].startswith('/')})
assert mapped==['/usr/lib64/libpython3.12.so.1.0']
p=subprocess.run(['rpm','-qf','--qf','%{NAME}|%{VERSION}|%{RELEASE}|%{ARCH}\n',str(Path(sys.executable).resolve()),mapped[0]],capture_output=True,text=True,timeout=15)
record={'job_id':os.environ['SLURM_JOB_ID'],'node':platform.node(),'pins':pins,'mapped_libpython':mapped,'package_query':{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},'sysconfig':{k:sysconfig.get_config_var(k) for k in source['sysconfig']},'CPATH':str(headers),'python':platform.python_version(),'pointer_bytes':struct.calcsize('P'),'PyObject_bytes':object.__basicsize__}
write('node-header-identity-'+os.environ['SLURM_JOB_ID']+'.json',record)
assert p.returncode==0 and p.stdout.splitlines()==['python3.12|3.12.13|2.el8_10|x86_64','python3.12-libs|3.12.13|2.el8_10|x86_64']
assert record['sysconfig']==source['sysconfig'] and record['python']==proof['runtime_python']=='3.12.13'
assert record['pointer_bytes']==8 and record['PyObject_bytes']==16 and sys.hexversion==51121648
assert str(headers)==proof['CPATH'] and ':' not in str(headers) and headers.is_absolute()
write('node-header-verified-'+os.environ['SLURM_JOB_ID']+'.json',dict(record,status='passed',all_215_header_hashes_verified=True))
print(str(headers))
RSNA_HEADER_GUARD
)
export CPATH
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
"$runtime_python" - <<'RSNA_SINGLE_GPU_CONTEXT'
import json
import torch

assert torch.cuda.device_count() == 1
torch.cuda.set_device(0)
value = torch.ones(8, device='cuda')
assert float(value.sum().item()) == 8.0
torch.cuda.synchronize()
print(json.dumps({'status': 'passed', 'cuda_context': True, 'gpu_count': 1}), flush=True)
RSNA_SINGLE_GPU_CONTEXT
if (( stop_requested )); then exit 75; fi
runtime_json=$(/usr/bin/python3.12 scripts/newton/runtime_archive.py extract --archive /home/sa305415/rsna-knee/cache/cache256-v3-official-recovery-v1.tar --sha256 6acf433e16c4832503103dd045943bbafa01dfb65fde8f56d93911fe1caac338 --expected-files 48772 --expected-bytes 82794366615 --scratch-base /tmp --min-free-gib 20)
runtime_root=$(/usr/bin/python3.12 -c 'import json,sys; print(json.load(sys.stdin)["runtime_root"])' <<< "$runtime_json")
runtime_cache="$runtime_root/cache"
export TRITON_CACHE_DIR="$runtime_root/triton-cache"
if [[ -e "$TRITON_CACHE_DIR" || -L "$TRITON_CACHE_DIR" ]]; then exit 2; fi
mkdir -- "$TRITON_CACHE_DIR"
"$runtime_python" scripts/newton/preflight.py --device cuda --expected-gpus 1 --repo /home/sa305415/rsna-knee/code-adapter-a32-all-8911f90 --expected-git 8911f90910ab8892b8324f6985c18d3ab146ec0c --cache "$runtime_cache" --cache-manifest /home/sa305415/rsna-knee/code-stopfix-1f486765/initiative/audits/full-image-recovery-v1/audit/series_cache_index.parquet --storage /home/sa305415/rsna-knee/logs --min-free-gb 20 --config /home/sa305415/rsna-knee/production-launch-adapter-a32-all-8911f90-v1/pilot-config.json
if (( stop_requested )); then exit 75; fi
IFS="," read -r -a assigned_gpus <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#assigned_gpus[@]} != 1 )); then exit 2; fi
"$runtime_python" scripts/newton/supervise.py --grace-seconds 120 --checkpoint /home/sa305415/rsna-knee/runs/dinov3-l16-adapter-a32-all-8911f90-pilot-v2/checkpoint.pt -- "$runtime_python" -m torch.distributed.run --rdzv-backend=c10d --rdzv-endpoint=127.0.0.1:0 --nnodes=1 --nproc-per-node=1 --max-restarts=0 --module rsna.competitive.train --studies /home/sa305415/rsna-knee/code-stopfix-1f486765/initiative/releases/report-supervision-v1/studies.parquet --series /home/sa305415/rsna-knee/code-stopfix-1f486765/initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv --cache "$runtime_cache" --folds /home/sa305415/rsna-knee/code-stopfix-1f486765/initiative/experiments/competitive-v1/folds --artifact-root /home/sa305415/rsna-knee/code-stopfix-1f486765 --config /home/sa305415/rsna-knee/production-launch-adapter-a32-all-8911f90-v1/pilot-config.json --output /home/sa305415/rsna-knee/runs/dinov3-l16-adapter-a32-all-8911f90-pilot-v2 --device cuda &
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
