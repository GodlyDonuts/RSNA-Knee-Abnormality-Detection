"""Preserve only original expanded2592 epochs; never launch/resume a GPU."""
import hashlib,json,shlex,subprocess,sys,time
from pathlib import Path
L=Path(__file__).resolve().parent;S='/tmp/rsna-research-20261001-stable.sock';started=time.monotonic();verified=[]
P=Path('scripts/hf/dispatch_compact_expanded_epoch.py');publisher_sha=hashlib.sha256(P.read_bytes()).hexdigest()
with (L/'reconnected-expanded2592-publication-watcher.intent.json').open('x') as f:json.dump({'original_job':json.loads((L/'GPU.submitted.json').read_text())['job_id'],'new_GPU_jobs':0,'maximum_seconds':57600,'publisher_coordinator_sha256':publisher_sha,'automatic_retry':False,'PID':__import__('os').getpid()},f)

def remote(args,timeout=60):
 x=subprocess.run(['ssh','-S',S,'-o','BatchMode=yes','newton',shlex.join(args)],capture_output=True,text=True,timeout=timeout)
 if x.returncode:raise RuntimeError('Original-node observation unavailable; no state inferred or GPU mutation')
 return x.stdout

def save_progress(phase,**fields):
 (L/'reconnected-expanded2592-publication-watcher-progress.json').write_text(json.dumps({'phase':phase,'verified_epochs':verified,'elapsed_seconds':time.monotonic()-started,**fields},indent=2)+'\n')

try:
 job=json.loads((L/'GPU.submitted.json').read_text())['job_id']
 while time.monotonic()-started<57600:
  states=remote(['sacct','-X','-j',job,'-n','-P','--format=State,ExitCode']).splitlines();states=[x for x in states if x.strip()];assert len(states)==1;state,exitcode=states[0].split('|')[:2]
  if state in ['PENDING','CONFIGURING','COMPLETING']:
   save_progress('actual_continuation_wait',job_id=job,state=state);time.sleep(60);continue
  scan="import json;from pathlib import Path;print(json.dumps(sorted(int(p.stem.split('-')[1]) for p in Path('/tmp/rsna-compact-expanded2592-production-20261016-v1/production/epoch-checkpoints').glob('epoch-*.json'))))"
  if state=='RUNNING':cmd=['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none','/usr/bin/python3.12','-B','-c',scan]
  else:cmd=['srun','--account=skattel','--partition=normal','--nodes=1','--nodelist=evc33','--ntasks=1','--cpus-per-task=1','--gres=none','--mem=2G','--time=00:02:00','--job-name=rsna-compact-epoch-terminal-scan','/usr/bin/python3.12','-B','-c',scan]
  epochs=json.loads(remote(cmd,180));assert all(type(e) is int and 1<=e<=16 for e in epochs)
  for epoch in epochs:
   intent=Path('artifacts/compact-native-n3-launch-20261016-v1')/f'expanded2592-publication/epoch-{epoch:03d}-intent.json';result=Path('artifacts/compact-native-n3-launch-20261016-v1')/f'expanded2592-publication/epoch-{epoch:03d}-result.json'
   if result.exists():
    x=json.loads(result.read_text());assert x['returncode']==0 and json.loads(x['stdout'])['status']=='private_epoch_fully_verified'
   elif intent.exists():raise RuntimeError('Existing epoch publication intent without verified result; reconcile read-only')
   else:
    assert hashlib.sha256(P.read_bytes()).hexdigest()==publisher_sha
    x=subprocess.run([sys.executable,str(P),str(epoch),'--ssh-control',S],capture_output=True,text=True,timeout=2000)
    with (L/f'reconnected-expanded2592-publication-watcher-epoch-{epoch:03d}.json').open('x') as f:json.dump({'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr},f,indent=2)
    assert x.returncode==0;value=json.loads(result.read_text());assert value['returncode']==0 and json.loads(value['stdout'])['status']=='private_epoch_fully_verified'
   if epoch not in verified:verified.append(epoch)
  save_progress('preserving_actual_original_epochs',job_id=job,state=state)
  if len(verified)==16:
   result={'status':'all16_original_epochs_privately_verified','job_id':job,'verified_epochs':verified,'new_GPU_jobs':0};break
  if state!='RUNNING':raise RuntimeError('Actual continuation terminal before all16 snapshots; completed snapshots preserved, exact recovery review required')
  time.sleep(60)
 else:raise TimeoutError('Finite handoff publication deadline')
except Exception as exc:result={'status':'stopped_for_exact_original_review','error_type':type(exc).__name__,'reason':str(exc),'verified_epochs':verified,'new_GPU_jobs':0,'automatic_retry':False}
(L/'reconnected-expanded2592-publication-watcher-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
