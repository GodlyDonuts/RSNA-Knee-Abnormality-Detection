#!/bin/bash
#SBATCH --job-name=rsna-git-status-diagnostic-v1
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
cd -- /home/sa305415/rsna-knee/code-stopfix-1f486765
export GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.preloadIndex GIT_CONFIG_VALUE_0=false
export PYTHONDONTWRITEBYTECODE=1
/usr/bin/python3.12 - <<'RSNA_GIT_DIAGNOSTIC'
import json,subprocess,time,os
start=time.monotonic()
try:
 result=subprocess.run(['git','status','--porcelain','--untracked-files=normal'],cwd='/home/sa305415/rsna-knee/code-stopfix-1f486765',capture_output=True,text=True,timeout=180)
 print(json.dumps({'status':'measured','seconds':round(time.monotonic()-start,3),'returncode':result.returncode,'stdout_bytes':len(result.stdout),'stderr_tail':result.stderr[-300:],'stdout_empty':not result.stdout.strip(),'job_id':os.environ.get('SLURM_JOB_ID'),'node':os.uname().nodename}),flush=True)
except subprocess.TimeoutExpired:
 print(json.dumps({'status':'timed_out_180','seconds':round(time.monotonic()-start,3),'job_id':os.environ.get('SLURM_JOB_ID'),'node':os.uname().nodename}),flush=True)
RSNA_GIT_DIAGNOSTIC
