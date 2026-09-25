#!/bin/bash
#SBATCH --job-name=rsna-f2e8-transfersyntax-transport
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-}" ]] || exit 2
root=/home/sa305415/rsna-knee
launch=$root/kaggle/launches/transfersyntax-e8-v1
projection=$root/kaggle/code-projections/e4bc973-b740-transfersyntax-v1
bundle=$root/kaggle/bundles/fold2-epoch008-eb280864bd1c-transfersyntax-v1
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
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
"$environment_root/environment/bin/python" -B "$launch/source/scripts/kaggle/coordinator.py" \
  prepare-transport \
  --projection "$launch/source" \
  --projection-manifest "$projection/projection.json" \
  --projection-sha256 3649c89d8a43b2bda9f8d59b1e59e3a8ff1e600516d33e6a88678b45a442ceea \
  --operation "$bundle/operation.json" \
  --snapshot "$launch/snapshot/epoch-008.json" \
  --publication-revision
