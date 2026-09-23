#!/bin/bash
#SBATCH --job-name=rsna-update23-cpu-probe
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd -- /home/sa305415/rsna-knee/code-stopfix-1f486765
test "$(git rev-parse HEAD)" = c58090e7c7a238461a5ac69293b32b64b74c6721
environment_root=
cleanup() {
  local code="$?"
  if [[ -n "$environment_root" ]]; then
    /usr/bin/python3.12 scripts/newton/runtime_environment.py cleanup --root "$environment_root" || code=2
  fi
  return "$code"
}
trap cleanup EXIT
environment_json=$(/usr/bin/python3.12 scripts/newton/runtime_environment.py extract \
  --archive /home/sa305415/rsna-knee/kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar \
  --sha256 89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338 \
  --expected-files 40403 --expected-bytes 6322691335 --scratch-base /tmp \
  --shared-python /usr/bin/python3.12 --min-free-gib 20)
environment_root=$(/usr/bin/python3.12 -c 'import json,sys; print(json.load(sys.stdin)["runtime_root"])' <<< "$environment_json")
"$environment_root/environment/bin/python" /home/sa305415/rsna-knee/diagnostics/update23-v1/extract_and_validate.py
