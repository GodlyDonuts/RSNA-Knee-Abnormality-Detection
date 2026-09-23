#!/bin/bash
#SBATCH --job-name=rsna-full-cache-scratch-probe-v1
#SBATCH --account=skattel
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --nodelist=evc49
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:05:00
#SBATCH --output=/home/sa305415/rsna-knee/logs/%x-%j.out
#SBATCH --error=/home/sa305415/rsna-knee/logs/%x-%j.err
set -euo pipefail
umask 077
/usr/bin/python3.12 - <<'RSNA_SCRATCH'
import json,os,shutil,subprocess,sys
from pathlib import Path
root=Path('/home/sa305415/rsna-knee')
archive=root/'cache/cache256-v3-official-recovery-v1.tar'
usage=shutil.disk_usage('/tmp')
mount=subprocess.run(['df','-T','/tmp'],capture_output=True,text=True,timeout=10)
need=82794366615+20*1024**3
record={'status':'passed' if archive.is_file() and archive.stat().st_size==82834493440 and usage.free>need else 'blocked','job_id':os.environ.get('SLURM_JOB_ID'),'node':os.uname().nodename,'cpus':os.environ.get('SLURM_CPUS_PER_TASK'),'gpu_allocation':os.environ.get('SLURM_JOB_GPUS'),'tmp_total_bytes':usage.total,'tmp_free_bytes':usage.free,'required_unpacked_plus_reserve_bytes':need,'archive_bytes':archive.stat().st_size if archive.exists() else None,'tmp_mount':mount.stdout[-1000:]}
print(json.dumps(record),flush=True)
sys.exit(0 if record['status']=='passed' else 2)
RSNA_SCRATCH
