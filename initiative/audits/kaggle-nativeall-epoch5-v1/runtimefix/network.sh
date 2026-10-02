#!/bin/bash
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-}" && "$(hostname)" == evc31 ]] || exit 2
root=/home/sa305415/rsna-knee
control=/tmp/rsna-kaggle-nativeall-epoch5-py312-v1
projection=/tmp/rsna-kaggle-nativeall-epoch5-py312-v1/projection/code
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
phase="${1:?phase}"
case "$phase" in push-runtime-corrected-kernel|submit) ;; *) exit 2 ;; esac
snapshot=/tmp/rsna-native384-all-production-20261002-v1/epoch-checkpoints/epoch-005.json
bundle=/tmp/rsna-kaggle-nativeall-epoch5-py312-v1/candidate
args=(--node-local-upload-root /tmp/rsna-kaggle-upload-85bfc1ce420f --projection "$projection" --projection-manifest "$control/projection/projection.json" --projection-sha256 a5e80083df543c3b9b7063dbfaa4741277dbe578e5d0a4fc679bb3350685a437 --operation "$bundle/operation.json" --snapshot "$snapshot" --dataset-publication-operation /tmp/rsna-kaggle-nativeall-epoch5-v3/native/operation.json)
if [[ "$phase" == push-runtime-corrected-kernel ]]; then args+=(--superseded-operation-sha256 3ba38976ab6f5a88f25cf54c983f62f53bae7137a3fb0bf0ba03eb006624e9f4); fi
"$python" "$projection/scripts/kaggle/coordinator.py" "$phase" "${args[@]}"
