#!/bin/bash
#SBATCH --job-name=rsna-f2e8-inputguard-bundle
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-}" ]] || exit 2
root=/home/sa305415/rsna-knee
projection=$root/kaggle/code-projections/f4f90bc-b740-inputguard-v1/code
bundle=$root/kaggle/bundles/fold2-epoch008-eb280864bd1c-inputguard-v1
snapshot=$root/kaggle/launches/transfersyntax-e8-v1/snapshot/epoch-008.json
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$projection
cd "$root/code-stopfix-1f486765"
environment_root=
cleanup() {
  local prior="$?"
  if [[ -n "$environment_root" ]]; then
    /usr/bin/python3.12 scripts/newton/runtime_environment.py cleanup --root "$environment_root" || prior=2
  fi
  return "$prior"
}
trap cleanup EXIT
record=$(/usr/bin/python3.12 scripts/newton/runtime_environment.py extract --archive "$root/kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar" --sha256 89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338 --expected-files 40403 --expected-bytes 6322691335 --scratch-base /tmp --shared-python /usr/bin/python3.12 --min-free-gib 20)
environment_root=$(/usr/bin/python3.12 -c 'import json,sys; print(json.load(sys.stdin)["runtime_root"])' <<< "$record")
python=$environment_root/environment/bin/python
"$python" "$projection/scripts/kaggle/workflow.py" prepare \
  --snapshot "$snapshot" \
  --encoder-spec "$root/kaggle/preparation-stage-v1/encoder-spec.json" \
  --wheelhouse "$root/kaggle/wheelhouse-cp312-cu130-v1" \
  --shared-wheelhouse "$root/kaggle/preparation-stage-v1/shared-wheelhouse.json" \
  --license-file "$root/encoder-stage/configs/competitive/weights/dinov3-large.LICENSE.md" \
  --license-source https://github.com/facebookresearch/dinov3/blob/ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md \
  --license-sha256 25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e \
  --output "$bundle" \
  --dataset-slug godlydonuts/rsna-knee-f2e008-inputguard-eb280864bd1c \
  --kernel-slug godlydonuts/rsna-knee-epoch8-f2-inputguard-eb280864bd1c
"$python" "$projection/scripts/kaggle/workflow.py" validate "$bundle/operation.json"
