#!/bin/bash
#SBATCH --job-name=rsna-epoch1-bundle
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=evc42
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=/home/sa305415/rsna-knee/kaggle/code-projections/2bc2695-b740-epoch1-v1/source
cd /home/sa305415/rsna-knee/code-stopfix-1f486765
environment_root=
cleanup() {
  local prior="$?"
  if [[ -n "$environment_root" ]]; then
    /usr/bin/python3.12 scripts/newton/runtime_environment.py cleanup --root "$environment_root" || prior=2
  fi
  return "$prior"
}
trap cleanup EXIT
record=$(/usr/bin/python3.12 scripts/newton/runtime_environment.py extract --archive /home/sa305415/rsna-knee/kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar --sha256 89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338 --expected-files 40403 --expected-bytes 6322691335 --scratch-base /tmp --shared-python /usr/bin/python3.12 --min-free-gib 20)
environment_root=$(/usr/bin/python3.12 -c 'import json,sys; print(json.load(sys.stdin)["runtime_root"])' <<< "$record")
"$environment_root/environment/bin/python" -c 'import importlib.util,pathlib,json
p=pathlib.Path
spec=importlib.util.spec_from_file_location("epoch_workflow",p('"'"'/home/sa305415/rsna-knee/kaggle/code-projections/2bc2695-b740-epoch1-v1/source'"'"')/"scripts/kaggle/workflow.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sidecar=p('"'"'/home/sa305415/rsna-knee/runs/dinov3-l16-epoch1-baseline-b740-v1/epoch-checkpoints/epoch-001.json'"'"')
print("'"'"'/home/sa305415/rsna-knee/runs/dinov3-l16-epoch1-baseline-b740-v1/epoch-checkpoints/epoch-001.json'"'"'-VALIDATION",m.validate_snapshot(sidecar,p('"'"'/home/sa305415/rsna-knee/kaggle/code-projections/2bc2695-b740-epoch1-v1/source'"'"'))[0]["status"],flush=True)
result=m.prepare_bundle(sidecar,p('"'"'/home/sa305415/rsna-knee/kaggle/preparation-stage-v1/encoder-spec.json'"'"'),p('"'"'/home/sa305415/rsna-knee/kaggle/wheelhouse-cp312-cu130-v1'"'"'),p('"'"'/home/sa305415/rsna-knee/kaggle/bundles/epoch001-01d56458c54f-2bc2695-v1'"'"'),"godlydonuts/rsna-knee-e001-01d56458c54f","godlydonuts/rsna-knee-e001-01d56458c54f",p('"'"'/home/sa305415/rsna-knee/encoder-stage/configs/competitive/weights/dinov3-large.LICENSE.md'"'"'),"https://github.com/facebookresearch/dinov3/blob/ffb4bb89c6558ca3244655c25a3955d01788b732/'"'"'/home/sa305415/rsna-knee/encoder-stage/configs/competitive/weights/dinov3-large.LICENSE.md'"'"'.md","25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e",repo=p('"'"'/home/sa305415/rsna-knee/kaggle/code-projections/2bc2695-b740-epoch1-v1/source'"'"'),shared_wheelhouse=p('"'"'/home/sa305415/rsna-knee/kaggle/preparation-stage-v1/shared-wheelhouse.json'"'"'))
print(json.dumps(result,sort_keys=True),flush=True)
'
