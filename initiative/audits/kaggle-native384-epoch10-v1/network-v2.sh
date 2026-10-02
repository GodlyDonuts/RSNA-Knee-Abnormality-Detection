#!/bin/bash
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-}" && "$(hostname)" == evc32 ]] || exit 2
root=/home/sa305415/rsna-knee
control=/home/sa305415/rsna-knee/kaggle/native384-epoch10-v1
projection=/home/sa305415/rsna-knee/kaggle/native384-epoch10-v1/projection/code
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
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
role="${1:?role}";phase="${2:?phase}"
case "$role" in native) ;; *) exit 2 ;; esac
case "$phase" in create-dataset|reconcile-dataset|push-kernel|submit) ;; *) exit 2 ;; esac
fields=$(/usr/bin/python3.12 -c 'import json,sys;v=json.load(open("/home/sa305415/rsna-knee/kaggle/native384-epoch10-v1/candidates-v2.json"))[sys.argv[1]];print(v["snapshot"]);print(v["bundle"])' "$role")
snapshot=$(sed -n '1p' <<< "$fields")
bundle=$(sed -n '2p' <<< "$fields")
args=(--projection "$projection" --projection-manifest "$control/projection/projection.json" --projection-sha256 688e5514a75fab67269ae44dfc9389847a68762b16e9cf77e5e4c009be40370c --operation "$bundle/operation.json" --snapshot "$snapshot")
if [[ "$phase" == create-dataset ]]; then args+=(--management-python "$root/kaggle/management-cp312-v1/bin/python"); fi
"$python" "$projection/scripts/kaggle/coordinator.py" "$phase" "${args[@]}"
