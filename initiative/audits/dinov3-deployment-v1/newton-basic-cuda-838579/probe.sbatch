#!/bin/bash
#SBATCH --job-name=rsna-gpu-visibility-v1
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100_pcie:2
#SBATCH --exclude=evc38
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:03:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/rsna-gpu-visibility-v1-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/rsna-gpu-visibility-v1-%j.err
set -euo pipefail
umask 077
cd /home/sa305415/rsna-knee/code
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset PYTHONPATH
environment_root=
cleanup() {
  local status=$?
  if [[ -n "$environment_root" ]]; then /usr/bin/python3.12 scripts/newton/runtime_environment.py cleanup --root "$environment_root" || status=2; fi
  return "$status"
}
trap cleanup EXIT
trap 'exit 2' TERM INT USR1
environment_json=$(/usr/bin/python3.12 scripts/newton/runtime_environment.py extract --archive /home/sa305415/rsna-knee/kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar --sha256 89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338 --expected-files 40403 --expected-bytes 6322691335 --scratch-base /tmp --shared-python /usr/bin/python3.12 --min-free-gib 20)
environment_root=$(/usr/bin/python3.12 -c 'import json,sys;print(json.load(sys.stdin)["runtime_root"])' <<< "$environment_json")
"$environment_root/environment/bin/python" /home/sa305415/rsna-knee/gpu-visibility-probe-v1/gpu_probe.py --phase batch
srun --ntasks=1 --cpus-per-task=4 --gres=gpu:nvidia_h100_pcie:2 "$environment_root/environment/bin/python" /home/sa305415/rsna-knee/gpu-visibility-probe-v1/gpu_probe.py --phase step
