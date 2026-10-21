"""Finite root continuation: one known C0 transport repair, only after actual failure.

No training, publication, Kaggle mutation or automatic retry of an uncertain
operation. The original owners remain responsible for every primary phase.
"""
import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / 'initiative/research/predictive-evc27-deadline-placement-20261020-v1'
ORIGINAL = ROOT / 'artifacts' / OLD.name / 'finite-owner'
JOURNAL = ROOT / 'artifacts' / HERE.name / 'C0-continuation'

OBSERVER = '''import json,os,hashlib,pathlib,shutil,stat,subprocess,sys
from datetime import datetime,timezone
v=json.load(sys.stdin)
assert os.uname().nodename=='evc33' and not os.environ.get('SLURM_JOB_GPUS')
p=pathlib.Path('/tmp/rsna-predictive-C0-evc27-return-20261020-v1')
names=['return-intent.json','publication/epoch-006/manifest.json']
assert p.is_dir() and not p.is_symlink()
assert sorted(str(f.relative_to(p)) for f in p.rglob('*') if f.is_file())==sorted(names)
files={}
for name in names:
 f=p/name;s=f.lstat()
 assert not f.is_symlink() and s.st_uid==os.getuid() and stat.S_IMODE(s.st_mode)==0o600 and s.st_size<1024*1024
 raw=f.read_bytes();files[name]={'bytes':len(raw),'inode':s.st_ino,'mode':'0o600','sha256':hashlib.sha256(raw).hexdigest()}
 if name=='return-intent.json':files[name]['intent']=json.loads(raw)
receiver=files['return-intent.json']['intent']['CPU_job']
assert str(receiver).isdigit() and v['sender'].isdigit()
sacct=subprocess.check_output(['env','TZ=UTC','sacct','-j',str(receiver)+','+v['sender'],'-X','-n','-P','-o','JobID,State,ExitCode,Elapsed,NodeList'],text=True).strip().splitlines()
for identity,node in [(str(receiver),'evc33'),(v['sender'],'evc27')]:
 assert any(r.startswith(identity+'|FAILED|1:0|') and r.endswith('|'+node) for r in sacct)
quota=subprocess.check_output(['lfs','quota','-u','sa305415','/home/sa305415'],text=True)
row=next(r.split() for r in quota.splitlines() if r.strip().startswith('/home/sa305415'))
paid=0
for arm,job in [('C','989035'),('B','989036'),('C0','989037')]:
 for suffix in ['out','err']:
  f=pathlib.Path('/home/sa305415/rsna-knee/kaggle/rsna-predictive-full2-'+arm+'-'+job+'.'+suffix);s=f.lstat()
  assert not f.is_symlink() and s.st_uid==os.getuid();paid+=1
remaining_bytes=(int(row[2])-int(row[1]))*1024-123*2**20
remaining_files=int(row[6])-int(row[5])-(376-paid)
assert remaining_bytes>20*2**30 and remaining_files>=5000
free=shutil.disk_usage('/tmp').free
assert free>128*2**30+512*2**20
print(json.dumps({'status':'actual_failed_C0_return_readonly_inspection','CPU_job':os.environ['SLURM_JOB_ID'],'UTC':datetime.now(timezone.utc).isoformat(),'files':files,'free_tmp_bytes':free,'mutations':0,'root_exists':True,'sacct':sacct,'raw_shared_quota':quota,'remaining_bytes_after_existing_liabilities':remaining_bytes,'remaining_files_after_existing_liabilities':remaining_files,'new_shared_remote_files':0,'existing_node_return_cap_bytes':512*2**20}))
'''


def read(path):
    return json.loads(path.read_bytes())


def once(path, value):
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write('\n')


def main(pin, socket):
    os.umask(0o077)
    for name in ('continue_C0.py','run_once.py','transport.py'):
        p=HERE/name
        assert p.read_bytes()==subprocess.check_output(['git','show',pin+':'+p.relative_to(ROOT).as_posix()],cwd=ROOT)
    JOURNAL.mkdir(parents=True,exist_ok=True,mode=0o700)
    once(JOURNAL/'owner-intent.json',{'source_git':pin,'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),
         'known_failure_only':True,'automatic_mutation_retry':False,'training_or_Kaggle_mutation':False})
    deadline=datetime(2026,10,22,10,tzinfo=timezone.utc).timestamp()
    while time.time()<deadline:
        if (OLD/'actual-owner-return-C0.json').exists():
            once(JOURNAL/'already-returned.json',{'status':'canonical_return_exists_no_action'});return
        if (OLD/'actual-return-unresolved-C0.json').exists() and (ORIGINAL/'return-unresolved-C0.json').exists():
            break
        time.sleep(60)
    else:
        raise TimeoutError('No eligible known C0 return failure within deadline')
    original=read(OLD/'actual-return-unresolved-C0.json')
    assert original==read(ORIGINAL/'return-unresolved-C0.json') and original['automatic_retry'] is False
    failed=read(ORIGINAL/'return-C0-sender-transport.json');assert failed['returncode']!=0
    senders=re.findall(r'srun: job (\d+) has been allocated resources',failed['stderr']);assert len(senders)==1
    once(JOURNAL/'destination-observation-intent.json',{'source_git':pin,'program_sha256':hashlib.sha256(OBSERVER.encode()).hexdigest()})
    command="import io,json,sys;e=json.load(sys.stdin);sys.stdin=io.StringIO(json.dumps(e['payload']));exec(compile(e['program'],'read_only_destination','exec'))"
    remote=['srun','--account=skattel','--partition=normal','--qos=normal','--nodelist=evc33',
            '--ntasks=1','--cpus-per-task=1','--gres=none','--mem=1G','--time=00:05:00',
            '--job-name=rsna-C0-return-observe','env','CUDA_VISIBLE_DEVICES=',
            '/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python','-B','-c',command]
    result=subprocess.run(['ssh','-S',socket,'-o','BatchMode=yes','-o','ConnectTimeout=8','newton',shlex.join(remote)],
        input=json.dumps({'program':OBSERVER,'payload':{'sender':senders[0]}}),text=True,capture_output=True,timeout=900)
    once(JOURNAL/'destination-observation-transport.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
    assert result.returncode==0,'Observation unavailable; no mutation performed'
    observed=json.loads(result.stdout);once(HERE/'actual-C0-failed-return-readonly.json',observed)
    for phase in ('inspect','repair'):
        if phase=='repair':
            proof=read(HERE/'actual-C0-source-reconciled.json')
            assert proof['status']=='actual_original_source_and_failed_prefix_reconciled' and proof['mutations']==0
            once(HERE/'actual-C0-repair-admission.json',{'UTC':datetime.now(timezone.utc).isoformat(),
                'root_admits_one_corrected_C0_return':True,'source_git':pin,
                'exact_failed_prefix_reconciled':True,'original_failed_partial_retained':True,
                'new_training_or_Kaggle_actions':0,**{k:observed[k] for k in ('existing_node_return_cap_bytes',
                'new_shared_remote_files','remaining_bytes_after_existing_liabilities','remaining_files_after_existing_liabilities','raw_shared_quota')}})
        with (JOURNAL/(phase+'.out')).open('x') as out,(JOURNAL/(phase+'.err')).open('x') as err:
            subprocess.run(['python3','-B',str(HERE/'run_once.py'),phase,'--arm','C0','--source-git',pin,'--socket',socket],cwd=ROOT,stdout=out,stderr=err,check=True,timeout=2400)
    final=read(OLD/'actual-owner-return-C0.json')
    assert final==read(HERE/'actual-C0-recovered-return.json')
    once(JOURNAL/'complete.json',{'status':'C0_known_order_failure_repaired_once','source_git':pin,
                                'actual_return':final,'UTC':datetime.now(timezone.utc).isoformat()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-git',required=True);p.add_argument('--socket',required=True)
    a=p.parse_args();main(a.source_git,a.socket)
