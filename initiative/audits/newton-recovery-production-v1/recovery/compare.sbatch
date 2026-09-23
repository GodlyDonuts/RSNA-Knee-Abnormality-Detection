#!/bin/bash
#SBATCH --job-name=rsna-signal-compare-v7
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=evc49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
cd -- /home/sa305415/rsna-knee/code-stopfix-1f486765
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
unset PYTHONPATH
actual_head=$(git rev-parse HEAD)
[[ "$actual_head" == c58090e7c7a238461a5ac69293b32b64b74c6721 ]]
environment_root=
cleanup_environment() {
  local previous_status="$?"
  if [[ -n "$environment_root" ]]; then
    /usr/bin/python3.12 scripts/newton/runtime_environment.py cleanup --root "$environment_root" || previous_status=2
  fi
  return "$previous_status"
}
trap cleanup_environment EXIT
environment_json=$(/usr/bin/python3.12 scripts/newton/runtime_environment.py extract --archive /home/sa305415/rsna-knee/kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar --sha256 89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338 --expected-files 40403 --expected-bytes 6322691335 --scratch-base /tmp --shared-python /usr/bin/python3.12 --min-free-gib 20)
environment_root=$(/usr/bin/python3.12 -c 'import json,sys; print(json.load(sys.stdin)["runtime_root"])' <<< "$environment_json")
"$environment_root/environment/bin/python" - <<'RSNA_COMPARE'
import hashlib,json,os
from pathlib import Path
import torch
root=Path('/home/sa305415/rsna-knee')
old=root/'runs/dinov3-l16-pilot100-fold0-e7b08625-v6/continuous'
new=root/'runs/dinov3-l16-signal-pilot100-c58090e-v7/resume'
control=root/'signal-rehearsal-v7';result_path=control/'comparison-result.json'
assert not result_path.exists()
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(2**20),b''):h.update(block)
 return h.hexdigest()
def equal(a,b):
 if type(a) is not type(b):return False
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
old_hash=sha(old/'checkpoint.pt');new_hash=sha(new/'checkpoint.pt')
assert old_hash=='b73fc313b432605edb860a07913707092363a9301e3db9f9cbd3685e1c2063dc'
assert new_hash=='905cac21f915a307aa21664b97bb2c1b1acbab2c657a920daa7f1cb26c45b3bc'
a=torch.load(old/'checkpoint.pt',map_location='cpu',weights_only=True,mmap=True)
b=torch.load(new/'checkpoint.pt',map_location='cpu',weights_only=True,mmap=True)
old_binding=a.pop('binding');new_binding=b.pop('binding')
old_code=old_binding.pop('code_sha256');new_code=new_binding.pop('code_sha256')
changed_code=sorted(k for k in set(old_code)|set(new_code) if old_code.get(k)!=new_code.get(k))
assert changed_code==['competitive/train.py']
binding_equal=equal(old_binding,new_binding)
payload_equal=equal(a,b)
old_pred=sha(old/'outer_predictions.csv');new_pred=sha(new/'outer_predictions.csv')
pred_equal=old_pred==new_pred=='66589afa68825daf32301ed48e93fde258b3d8a3829e167cf48dcc6c3553262a'
answer={'status':'passed_exact_payload_and_predictions' if binding_equal and payload_equal and pred_equal else 'comparison_failed','job_id':os.environ.get('SLURM_JOB_ID'),'cpu_only':True,'torch_version':torch.__version__,'old_checkpoint_sha256':old_hash,'new_checkpoint_sha256':new_hash,'old_prediction_sha256':old_pred,'new_prediction_sha256':new_pred,'changed_code_modules':changed_code,'binding_equal_excluding_code':binding_equal,'checkpoint_payload_elementwise_equal_excluding_binding':payload_equal,'prediction_bytes_equal':pred_equal,'optimizer_steps':b.get('step'),'old_epoch':a.get('epoch'),'new_epoch':b.get('epoch')}
with result_path.open('x') as f:json.dump(answer,f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(answer),flush=True)
if answer['status']!='passed_exact_payload_and_predictions':raise SystemExit(2)
RSNA_COMPARE
