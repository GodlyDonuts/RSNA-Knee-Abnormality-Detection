"""Single use local coordinator; an existing intent forbids automatic retry."""
import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]


def main():
    os.umask(0o077)
    pin=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    files={}
    for name in ('diagnostic.py','stage.py','dispatch_once.py'):
        p=HERE/name
        raw=subprocess.check_output(['git','show',pin+':'+str(p.relative_to(REPO))],cwd=REPO)
        assert raw==p.read_bytes()
        files[name]=raw.decode()
    journal=REPO/'artifacts'/HERE.name
    journal.mkdir(mode=0o700,exist_ok=False)
    payload=dict(source_git=pin,source=files['diagnostic.py'],sha256=hashlib.sha256(files['diagnostic.py'].encode()).hexdigest())
    with (journal/'intent.json').open('x') as f:
        json.dump(dict(source_git=pin,diagnostic_sha256=payload['sha256'],automatic_retry=False),f)
    p=REPO/'initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py'
    s=importlib.util.spec_from_file_location('transport',p); t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
    cmd=['srun','--account=skattel','--partition=normal','--qos=normal','--nodelist=evc27',
        '--ntasks=1','--cpus-per-task=1','--gres=none','--mem=1G','--time=00:05:00',
        '--job-name=rsna-C100-readout-stage','env','CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1',
        '/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python','-B','-c',t.COMMAND]
    x=subprocess.run(['ssh','-S','/tmp/rsna-research-20261001-stable.sock','-o','BatchMode=yes','newton',shlex.join(cmd)],
        input=t.envelope(files['stage.py'],payload),text=True,capture_output=True,timeout=600)
    with (journal/'transport.json').open('x') as f:
        json.dump(dict(returncode=x.returncode,stdout=x.stdout,stderr=x.stderr),f)
    assert x.returncode==0, 'Original intent retained; inspect uncertainty read only'
    result=json.loads(x.stdout)
    with (journal/'result.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
