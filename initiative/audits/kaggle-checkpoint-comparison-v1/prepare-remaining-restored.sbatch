#!/bin/bash
#SBATCH --job-name=rsna-f2-restored-prepare
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
projection=$root/kaggle/code-projections/f4f90bc-b740-inputguard-v1/code
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
"$python" - <<'DIAGNOSTIC'
import json,sys,traceback
from pathlib import Path
sys.path.insert(0,'/home/sa305415/rsna-knee/kaggle/code-projections/f4f90bc-b740-inputguard-v1/code/scripts/kaggle')
from workflow import validate_snapshot
snapshot=Path('/home/sa305415/rsna-knee/runs/dinov3-l16-fold2-b74060b-1gpu-production-v1/epoch-checkpoints/epoch-008.json')
try:
 meta,_,_=validate_snapshot(snapshot,Path('/home/sa305415/rsna-knee/kaggle/code-projections/f4f90bc-b740-inputguard-v1/code'))
 print(json.dumps({'phase':'epoch8_snapshot_validation','status':'passed','sha':meta['checkpoint_sha256']}),flush=True)
except Exception as exc:
 print(json.dumps({'phase':'epoch8_snapshot_validation','error_type':type(exc).__name__,'detail':str(exc)}),flush=True)
 traceback.print_exc()
 raise
DIAGNOSTIC

snapshot=/home/sa305415/rsna-knee/runs/dinov3-l16-fold2-b74060b-1gpu-production-v1/epoch-checkpoints/epoch-008.json
bundle=$root/kaggle/bundles/fold2-epoch008-eb280864bd1c-inputguard-v1
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

snapshot=/home/sa305415/rsna-knee/runs/dinov3-l16-fold2-b74060b-1gpu-production-v1/epoch-checkpoints/epoch-006.json
bundle=$root/kaggle/bundles/fold2-epoch006-532331e0fd06-inputguard-v1
"$python" "$projection/scripts/kaggle/workflow.py" prepare \
  --snapshot "$snapshot" \
  --encoder-spec "$root/kaggle/preparation-stage-v1/encoder-spec.json" \
  --wheelhouse "$root/kaggle/wheelhouse-cp312-cu130-v1" \
  --shared-wheelhouse "$root/kaggle/preparation-stage-v1/shared-wheelhouse.json" \
  --license-file "$root/encoder-stage/configs/competitive/weights/dinov3-large.LICENSE.md" \
  --license-source https://github.com/facebookresearch/dinov3/blob/ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md \
  --license-sha256 25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e \
  --output "$bundle" \
  --dataset-slug godlydonuts/rsna-knee-f2e006-inputguard-532331e0fd06 \
  --kernel-slug godlydonuts/rsna-knee-epoch6-f2-inputguard-532331e0fd06
"$python" "$projection/scripts/kaggle/workflow.py" validate "$bundle/operation.json"

snapshot=/home/sa305415/rsna-knee/runs/dinov3-l16-fold2-b74060b-1gpu-production-v1/epoch-checkpoints/epoch-004.json
bundle=$root/kaggle/bundles/fold2-epoch004-9f85d010fba4-inputguard-v1
"$python" "$projection/scripts/kaggle/workflow.py" prepare \
  --snapshot "$snapshot" \
  --encoder-spec "$root/kaggle/preparation-stage-v1/encoder-spec.json" \
  --wheelhouse "$root/kaggle/wheelhouse-cp312-cu130-v1" \
  --shared-wheelhouse "$root/kaggle/preparation-stage-v1/shared-wheelhouse.json" \
  --license-file "$root/encoder-stage/configs/competitive/weights/dinov3-large.LICENSE.md" \
  --license-source https://github.com/facebookresearch/dinov3/blob/ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md \
  --license-sha256 25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e \
  --output "$bundle" \
  --dataset-slug godlydonuts/rsna-knee-f2e004-inputguard-9f85d010fba4 \
  --kernel-slug godlydonuts/rsna-knee-epoch4-f2-inputguard-9f85d010fba4
"$python" "$projection/scripts/kaggle/workflow.py" validate "$bundle/operation.json"

snapshot=/home/sa305415/rsna-knee/runs/dinov3-l16-fold2-b74060b-1gpu-production-v1/epoch-checkpoints/epoch-002.json
bundle=$root/kaggle/bundles/fold2-epoch002-092c3eed133f-inputguard-v1
"$python" "$projection/scripts/kaggle/workflow.py" prepare \
  --snapshot "$snapshot" \
  --encoder-spec "$root/kaggle/preparation-stage-v1/encoder-spec.json" \
  --wheelhouse "$root/kaggle/wheelhouse-cp312-cu130-v1" \
  --shared-wheelhouse "$root/kaggle/preparation-stage-v1/shared-wheelhouse.json" \
  --license-file "$root/encoder-stage/configs/competitive/weights/dinov3-large.LICENSE.md" \
  --license-source https://github.com/facebookresearch/dinov3/blob/ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md \
  --license-sha256 25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e \
  --output "$bundle" \
  --dataset-slug godlydonuts/rsna-knee-f2e002-inputguard-092c3eed133f \
  --kernel-slug godlydonuts/rsna-knee-epoch2-f2-inputguard-092c3eed133f
"$python" "$projection/scripts/kaggle/workflow.py" validate "$bundle/operation.json"
