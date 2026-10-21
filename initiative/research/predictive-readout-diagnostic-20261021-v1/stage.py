"""One owning-node CPU stage/dispatch; no new inputs or checkpoint copies."""
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path('/tmp/rsna-predictive-readout-diagnostic-20261021-v1')
CONTROL=Path('/tmp/rsna-predictive-density100-full-20261020-v1')
RUN=Path('/tmp/rsna-predictive-full-C100-20261020-v1')


def once(p,v):
    with p.open('x') as f: json.dump(v,f,sort_keys=True,allow_nan=False)


def main(v):
    os.umask(0o077)
    assert os.uname().nodename=='evc27' and not os.environ.get('SLURM_JOB_GPUS')
    assert not ROOT.exists(), 'Existing diagnostic requires reconciliation'
    assert re.fullmatch('[a-f0-9]{40}',v['source_git'])
    assert datetime.now(timezone.utc) < datetime(2026,10,22,0,tzinfo=timezone.utc)
    assert hashlib.sha256(v['source'].encode()).hexdigest()==v['sha256']
    assert hashlib.sha256((CONTROL/'binding.private.json').read_bytes()).hexdigest()=='f0d73e6cc04cbf7b34f06499f5aef4ba60ea2c593662764b8f64ec5105a289cd'
    b=json.loads((CONTROL/'binding.private.json').read_text())
    t=json.loads((RUN/'terminal.json').read_text())
    assert t['updates']==2916 and t['epochs']==6 and t['GPU_job']=='995351'
    assert (RUN/'publication/epoch-006/model.pt').stat().st_size==293424153
    scheduler=subprocess.check_output(['sacct','-X','-j','995351','-n','-P','-o','JobID,State,ExitCode'],text=True)
    assert '995351|COMPLETED|0:0' in scheduler
    node=subprocess.check_output(['scontrol','show','node','evc27','-o'],text=True)
    assert 'gpu:nvidia_h100_pcie:2' in node and 'gres/gpu=2' in node
    quota=subprocess.check_output(['lfs','quota','-u','sa305415','/home/sa305415'],text=True)
    fields=next(x.split() for x in quota.splitlines() if x.strip().startswith('/home/sa305415'))
    shared_bytes=(int(fields[2])-int(fields[1]))*1024
    shared_files=int(fields[6])-int(fields[5])
    assert shared_bytes-125*2**20 > 20*2**30 and shared_files-370 >= 5000
    free=shutil.disk_usage('/tmp').free
    liability=518*2**30+(176+16)*2**20
    assert free-liability > 20*2**30
    ROOT.mkdir(mode=0o700)
    (ROOT/'diagnostic.py').write_text(v['source'])
    admission=dict(source_git=v['source_git'],diagnostic_sha256=v['sha256'],CPU_job=os.environ['SLURM_JOB_ID'],
        new_node_cap_bytes=16*2**20,node_free_bytes=free,node_liabilities_including_new=liability,
        shared_free_bytes=shared_bytes,shared_free_files=shared_files,
        shared_liability_bytes=125*2**20,shared_liability_files=370,new_shared_bytes=0,new_shared_files=0,
        original_model_or_input_bytes_copied=0,independent_accuracy_claim=False)
    once(ROOT/'admission.json',admission)
    script='\n'.join(['#!/bin/bash','#SBATCH --account=skattel','#SBATCH --partition=normal',
        '#SBATCH --qos=normal','#SBATCH --nodelist=evc27','#SBATCH --ntasks=1',
        '#SBATCH --cpus-per-task=2','#SBATCH --gres=gpu:nvidia_h100_pcie:1',
        '#SBATCH --mem=16G','#SBATCH --time=00:25:00','#SBATCH --deadline=2026-10-22T03:00:00',
        '#SBATCH --job-name=rsna-C100-readout-diagnostic',
        '#SBATCH --output='+str(ROOT/'GPU-%j.out'),'#SBATCH --error='+str(ROOT/'GPU-%j.err'),
        'set -euo pipefail','umask 077',
        'export TZ=UTC PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUBLAS_WORKSPACE_CONFIG=:4096:8 HF_HUB_OFFLINE=1',
        'exec '+b['runtime_python']+' -B '+str(ROOT/'diagnostic.py'),''])
    (ROOT/'GPU.sbatch').write_text(script)
    once(ROOT/'dispatch-intent.json',dict(source_git=v['source_git'],script_sha256=hashlib.sha256(script.encode()).hexdigest(),automatic_retry=False))
    x=subprocess.run(['sbatch','--parsable'],input=script,text=True,capture_output=True,timeout=60,
        env={**os.environ,'TZ':'UTC'})
    once(ROOT/'dispatch-transport.json',dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr))
    assert x.returncode==0 and re.fullmatch(r'\d+(?:;[^\s]+)?\s*',x.stdout), 'Reconcile uncertainty; no retry'
    result=dict(admission,status='fixed_final_readout_diagnostic_submitted_once',GPU_job=x.stdout.strip().split(';')[0],
        node='evc27',maximum_runtime_seconds=1500,studies=128,fitting_only=True)
    once(ROOT/'submitted.json',result)
    return result


if __name__=='__main__':
    import sys
    print(json.dumps(main(json.load(sys.stdin))),flush=True)
