#!/bin/bash
#SBATCH --job-name=rsna-a64-e10-prepfix2
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
projection=/home/sa305415/rsna-knee/kaggle/rank64-epoch10-v1/projection/code
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


"$python" - <<'RSNA_RANK64_MANIFEST_FIX'
import json,subprocess;from pathlib import Path;c=Path('/home/sa305415/rsna-knee/kaggle/rank64-epoch10-v1');assert subprocess.check_output(['sacct','-X','--noheader','--parsable2','-j','866324','--format=JobIDRaw,State,ExitCode'],text=True).strip()=='866324|FAILED|1:0';assert not Path('/home/sa305415/rsna-knee/kaggle/bundles/a64-epoch010-0f1a410916b6-v1').exists();assert json.loads((c/'acceptance.json').read_text())['projected_remaining_gib']>=20;assert not (c/'encoder-spec-fixed.json').exists();(c/'encoder-spec-fixed.json').write_text(json.dumps({'adapter': {'alpha': 8.0, 'dropout': 0.05, 'ffn_last_layers': 12, 'gradient_checkpointing': True, 'rank': 64}, 'architecture': 'dinov3_l16', 'config_path': '/home/sa305415/rsna-knee/encoder-stage/configs/competitive/weights/dinov3-large.config.json', 'config_sha256': '135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5', 'license': 'other', 'source_id': 'facebook/dinov3-vitl16-pretrain-lvd1689m', 'source_revision': 'ea8dc2863c51be0a264bab82070e3e8836b02d51', 'state_dict_format': 'safetensors', 'transformers_version': '5.6.1', 'weights_path': '/home/sa305415/rsna-knee/encoder-stage/artifacts/encoder-weights/dinov3-large/model.safetensors', 'weights_sha256': 'dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179'}));print('original rank64 training encoder manifest bound',flush=True)
RSNA_RANK64_MANIFEST_FIX
snapshot=/home/sa305415/rsna-knee/runs/dinov3-l16-adapter-a64-sampled-fold0-c9fcc7e-production-v1/epoch-checkpoints/epoch-010.json
bundle=/home/sa305415/rsna-knee/kaggle/bundles/a64-epoch010-0f1a410916b6-v1
"$python" "$projection/scripts/kaggle/workflow.py" prepare \
 --snapshot "$snapshot" --encoder-spec "/home/sa305415/rsna-knee/kaggle/rank64-epoch10-v1/encoder-spec-fixed.json" \
 --wheelhouse "$root/kaggle/wheelhouse-cp312-cu130-v1" --shared-wheelhouse "$root/kaggle/preparation-stage-v1/shared-wheelhouse.json" \
 --license-file "$root/encoder-stage/configs/competitive/weights/dinov3-large.LICENSE.md" \
 --license-source https://github.com/facebookresearch/dinov3/blob/ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md \
 --license-sha256 25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e \
 --output "$bundle" --dataset-slug godlydonuts/rsna-knee-a64-e010-0f1a410916b6 --kernel-slug godlydonuts/rsna-knee-a64-epoch10-0f1a410916b6
"$python" "$projection/scripts/kaggle/workflow.py" validate "$bundle/operation.json"
"$python" "$projection/scripts/kaggle/coordinator.py" prepare-transport \
 --projection "$projection" --projection-manifest "${projection%/code}/projection.json" \
 --projection-sha256 688e5514a75fab67269ae44dfc9389847a68762b16e9cf77e5e4c009be40370c --snapshot "$snapshot" --operation "$bundle/operation.json"
