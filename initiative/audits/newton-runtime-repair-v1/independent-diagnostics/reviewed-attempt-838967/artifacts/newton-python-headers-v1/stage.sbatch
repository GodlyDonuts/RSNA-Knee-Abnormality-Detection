#!/bin/bash
#SBATCH --job-name=rsna-python-headers-v1
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --constraint=h100
#SBATCH --exclude=evc38,evc50
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=none
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=/home/sa305415/rsna-knee/python-headers-3.12.13-v1/batch.stdout
#SBATCH --error=/home/sa305415/rsna-knee/python-headers-3.12.13-v1/batch.stderr
set -euo pipefail
umask 077
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
unset PYTHONPATH
/usr/bin/python3.12 /home/sa305415/rsna-knee/python-headers-3.12.13-v1/stage_and_verify.py stage
cd /home/sa305415/rsna-knee/code
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
export CPATH=/home/sa305415/rsna-knee/python-headers-3.12.13-v1/extracted/usr/include/python3.12
"$environment_root/environment/bin/python" /home/sa305415/rsna-knee/python-headers-3.12.13-v1/stage_and_verify.py proof
