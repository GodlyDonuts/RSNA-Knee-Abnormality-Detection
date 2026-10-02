#!/bin/bash
#SBATCH --job-name=rsna-nativeall-e5-prepare-v3
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=evc31
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
[[ -n "${SLURM_JOB_ID:-}" && -z "${SLURM_JOB_GPUS:-}" && "$(hostname)" == evc31 ]] || exit 2
root=/home/sa305415/rsna-knee
control=/home/sa305415/rsna-knee/kaggle/nativeall-epoch5-v1
projection=/tmp/rsna-kaggle-nativeall-epoch5-v3/projection/code
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
import hashlib,json,os,shutil,tarfile
from pathlib import Path
r=Path('/tmp/rsna-kaggle-nativeall-epoch5-v3');assert not r.exists();assert shutil.disk_usage('/tmp').free>=20*1024**3+2*1024**3;r.mkdir(mode=0o700)
c=Path('/home/sa305415/rsna-knee/kaggle/nativeall-epoch5-v1'); m=json.loads((c/'projection-v3.json').read_text()); assert hashlib.sha256((c/'projection-v3.json').read_bytes()).hexdigest()=='74cbc956e14a4b3f6e146834a06d96dfa2b65281202676df43e82993c5cdcb76'
code=r/'projection/code';code.mkdir(parents=True)
with tarfile.open(c/'projection-v3.tar') as t:
 members=t.getmembers();assert {x.name for x in members}==set(m['files']) and len(members)==len(m['files'])
 for item in members:
  assert item.isfile() and not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
  raw=t.extractfile(item).read();v=m['files'][item.name];assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
  p=code/item.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(v['mode'])

NODE_BUDGET
"$python" "$projection/scripts/kaggle/workflow.py" prepare --snapshot /tmp/rsna-native384-all-production-20261002-v1/epoch-checkpoints/epoch-005.json --encoder-spec "$control/encoder-spec-native-r32.json" --wheelhouse "$root/kaggle/wheelhouse-cp312-cu130-v1" --shared-wheelhouse "$root/kaggle/preparation-stage-v1/shared-wheelhouse.json" --license-file "$root/encoder-stage/configs/competitive/weights/dinov3-large.LICENSE.md" --license-source https://github.com/facebookresearch/dinov3/blob/ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md --license-sha256 25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e --output /tmp/rsna-kaggle-nativeall-epoch5-v3/native --dataset-slug godlydonuts/rsna-knee-nativeall-e005-85bfc1ce420f --kernel-slug godlydonuts/rsna-knee-nativeall-epoch5-85bfc1ce420f
"$python" "$projection/scripts/kaggle/workflow.py" validate /tmp/rsna-kaggle-nativeall-epoch5-v3/native/operation.json
"$python" "$projection/scripts/kaggle/coordinator.py" prepare-transport --projection "$projection" --projection-manifest "$control/projection-v3.json" --projection-sha256 74cbc956e14a4b3f6e146834a06d96dfa2b65281202676df43e82993c5cdcb76 --snapshot /tmp/rsna-native384-all-production-20261002-v1/epoch-checkpoints/epoch-005.json --node-local-upload-root /tmp/rsna-kaggle-upload-85bfc1ce420f --operation /tmp/rsna-kaggle-nativeall-epoch5-v3/native/operation.json
/usr/bin/python3.12 - <<'PREPARATION_RESULT'
import json,os
from pathlib import Path
c=Path('/home/sa305415/rsna-knee/kaggle/nativeall-epoch5-v1');op=json.loads(Path('/tmp/rsna-kaggle-nativeall-epoch5-v3/native/operation.json').read_text());j=Path('/home/sa305415/rsna-knee/kaggle/dataset-publications')/op['checkpoint_sha256'];t=json.loads((j/'transport.json').read_text());assert t['archive_bytes']<1350*1024**2 and t['node_local_upload']['host']=='evc31'
r={'status':'actual_full_data_native_epoch5_prepared_not_published','job_id':os.environ['SLURM_JOB_ID'],'operation':op,'transport':t,'bundle_node':'evc31','no_competition_submission':True}
with (c/'prepared-v3.json').open('x') as stream:json.dump(r,stream,indent=2)
print(json.dumps(r))
PREPARATION_RESULT
