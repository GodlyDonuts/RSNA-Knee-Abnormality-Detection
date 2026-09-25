#!/bin/bash
set -euo pipefail
umask 077
phase=${1:?phase required}
case "$phase" in
  prepare-transport|create-dataset|reconcile-dataset|push-after-failed-submission|submit) ;;
  *) exit 2 ;;
esac
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-}" ]] || exit 2
root=/home/sa305415/rsna-knee
projection=$root/kaggle/code-projections/94215dc-b740-studyfallback2-v1
bundle=$root/kaggle/bundles/fold2-epoch008-eb280864bd1c-studyfallback2-v1
snapshot=$root/runs/dinov3-l16-fold2-b74060b-1gpu-production-v1/epoch-checkpoints/epoch-008.json
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=$projection/source
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
args=("$phase" --projection "$projection/source" --projection-manifest "$projection/projection.json" --projection-sha256 0f2adcf19e46bb91478d27993cdbb3e402cb87d339252a9523434a5613e3c942 --operation "$bundle/operation.json" --snapshot "$snapshot")
args+=(--publication-revision)
if [[ "$phase" == push-after-failed-submission ]]; then
  args+=(--superseded-operation-sha256 f34779feeefceaba38e912c590af3770074db942a8a7744ec0c58bc978fe37a1)
fi
if [[ "$phase" == create-dataset ]]; then
  args+=(--management-python "$root/kaggle/management-cp312-v1/bin/python")
fi
"$environment_root/environment/bin/python" -B "$projection/source/scripts/kaggle/coordinator.py" "${args[@]}"
