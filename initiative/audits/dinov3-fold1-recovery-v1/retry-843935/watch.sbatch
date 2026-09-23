#!/bin/bash
#SBATCH --job-name=rsna-dinov3-fold1-archive-v2
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
cd /home/sa305415/rsna-knee/code-stopfix-1f486765
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
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
"$runtime_root/environment/bin/python" /home/sa305415/rsna-knee/production-launch-b740-fold1-v2/archive_odd_epochs.py --log /home/sa305415/rsna-knee/logs/rsna-dinov3-fold1-b740-v2-843935.out --output /home/sa305415/rsna-knee/runs/dinov3-l16-production-fold1-b74060b-v2 --primary-job 843935 --interval 2
