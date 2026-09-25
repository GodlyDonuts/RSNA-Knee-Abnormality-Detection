#!/bin/bash
# Run inside a one-CPU, zero-GPU srun allocation. JSON token arrives on stdin.
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-}" ]] || exit 2
phase="${1:?phase required}"
case "$phase" in
  create-dataset|reconcile-dataset|push-after-failed-submission|submit) ;;
  *) exit 2 ;;
esac
root=/home/sa305415/rsna-knee
projection=$root/kaggle/code-projections/90d879a-b740-metadata-v1
bundle=$root/kaggle/bundles/fold2-epoch008-eb280864bd1c-metadata-v1
snapshot=$root/kaggle/launches/transfersyntax-e8-v1/snapshot/epoch-008.json
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$projection/code
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
args=(
  --projection "$projection/code"
  --projection-manifest "$projection/projection.json"
  --projection-sha256 9f89f4bb28272f2c9829ed40c561a59a40d6b5273ac7429d8a81e34ac58d7616
  --operation "$bundle/operation.json"
  --snapshot "$snapshot"
  --publication-revision
)
case "$phase" in
  create-dataset)
    args+=(--management-python "$root/kaggle/management-cp312-v1/bin/python") ;;
  push-after-failed-submission)
    args+=(--superseded-operation-sha256 645ce5805c606b6fef2ca8975d9bcd896212fe22ccaff563e5bc41e95496a3d0) ;;
esac
"$python" "$projection/code/scripts/kaggle/coordinator.py" "$phase" "${args[@]}"
