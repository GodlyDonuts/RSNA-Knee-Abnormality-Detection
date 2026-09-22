#!/bin/bash
#SBATCH --job-name=rsna-dinov3-pilot-v1
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
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
unset PYTHONPATH
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
"$runtime_python" scripts/newton/preflight.py --device cuda --expected-gpus 2 --repo /home/sa305415/rsna-knee/code --expected-git 06816e6b698901bd5148d47a3cdf73252c5a3a89 --cache "$runtime_cache" --cache-manifest /home/sa305415/rsna-knee/code/initiative/audits/full-image-recovery-v1/audit/series_cache_index.parquet --storage /home/sa305415/rsna-knee/logs --min-free-gb 20 --config /home/sa305415/rsna-knee/encoder-stage/run-configs/pilot100-fold0.json --require-nccl
if (( stop_requested )); then exit 75; fi
IFS="," read -r -a assigned_gpus <<< "$CUDA_VISIBLE_DEVICES"
if (( ${#assigned_gpus[@]} != 2 )); then exit 2; fi
"$runtime_python" scripts/local_competitive_acceptance.py --cache "$runtime_cache" --output /home/sa305415/rsna-knee/runs/dinov3-l16-pilot100-fold0-06816e6b --world-size 2 --config /home/sa305415/rsna-knee/encoder-stage/run-configs/pilot100-fold0.json --device cuda --skip-public-examples &
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
