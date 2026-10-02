#!/bin/bash
#SBATCH --job-name=rsna-native384-e10-prepare
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=evc32
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
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
/usr/bin/python3.12 - <<'NODE_BUDGET'
import json,os,shutil
from pathlib import Path
r=Path('/tmp/rsna-kaggle-native384-epoch10-v1');assert not r.exists();assert shutil.disk_usage('/tmp').free>=20*1024**3+2*1024**3;r.mkdir(mode=0o700)
NODE_BUDGET
"$python" "$projection/scripts/kaggle/workflow.py" prepare --snapshot /home/sa305415/rsna-knee/runs/dinov3-l16-adapter-a32-native384-fold0-c9fcc7e-production-v1/epoch-checkpoints/epoch-010.json --encoder-spec "$control/encoder-spec.json" --wheelhouse "$root/kaggle/wheelhouse-cp312-cu130-v1" --shared-wheelhouse "$root/kaggle/preparation-stage-v1/shared-wheelhouse.json" --license-file "$root/encoder-stage/configs/competitive/weights/dinov3-large.LICENSE.md" --license-source https://github.com/facebookresearch/dinov3/blob/ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md --license-sha256 25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e --output /tmp/rsna-kaggle-native384-epoch10-v1/native --dataset-slug godlydonuts/rsna-knee-native384-e010-24cf7a0c936e --kernel-slug godlydonuts/rsna-knee-native384-epoch10-24cf7a0c936e
"$python" "$projection/scripts/kaggle/workflow.py" validate /tmp/rsna-kaggle-native384-epoch10-v1/native/operation.json
"$python" "$projection/scripts/kaggle/coordinator.py" prepare-transport --projection "$projection" --projection-manifest "$control/projection/projection.json" --projection-sha256 688e5514a75fab67269ae44dfc9389847a68762b16e9cf77e5e4c009be40370c --snapshot /home/sa305415/rsna-knee/runs/dinov3-l16-adapter-a32-native384-fold0-c9fcc7e-production-v1/epoch-checkpoints/epoch-010.json --operation /tmp/rsna-kaggle-native384-epoch10-v1/native/operation.json
/usr/bin/python3.12 - <<'PREPARATION_RESULT'
import json,os
from pathlib import Path
control=Path('/home/sa305415/rsna-knee/kaggle/native384-epoch10-v1');plan=json.loads((control/'candidates.json').read_text());rows={}
for role,c in plan.items():
 op=json.loads((Path(c['bundle'])/'operation.json').read_text());journal=Path('/home/sa305415/rsna-knee/kaggle/dataset-publications')/c['sha'];t=json.loads((journal/'transport.json').read_text());assert t['archive_bytes']<1350*1024**2;rows[role]={'operation':op,'transport':t,'bundle_node':'evc32'}
out={'status':'exact_native384_final_bundle_prepared_not_published','job_id':os.environ['SLURM_JOB_ID'],'candidates':rows,'no_competition_submission':True}
with (control/'prepared.json').open('x') as f:json.dump(out,f,indent=2)
print(json.dumps(out))
PREPARATION_RESULT
