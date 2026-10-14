import json,os,subprocess,sys,time
from pathlib import Path
L=Path('artifacts/final-public-source-fit-evc40-20261013-v1/epoch9-observer');assert not (L/'intent.json').exists()
with (L/'intent.json').open('x') as f:json.dump({'epoch':9,'PID':os.getpid(),'deadline_seconds':1800,'publication_coordinator':'original_publication_coordinator.py','automatic_publication_retry':False,'does_not_modify_training':True},f,indent=2)
ssh=['ssh','-S','/tmp/rsna-research-20261001-stable.sock','-o','BatchMode=yes','newton'];deadline=time.monotonic()+1800
try:
 while time.monotonic()<deadline:
  x=subprocess.run(ssh+["squeue -h -j 908454 -o '%T'"],capture_output=True,text=True,timeout=20)
  if x.returncode:raise RuntimeError('Observation lost; no mutation or restart')
  if x.stdout.strip()!='RUNNING':raise RuntimeError('Original allocation not running; reconcile actual original outcome')
  x=subprocess.run(ssh+["srun --jobid=908454 --overlap --ntasks=1 --cpus-per-task=1 --gres=none --mem=1G --time=00:01:00 /bin/sh -c 'if test -f /tmp/rsna-final-public-source-fit-evc40-20261013-v1/production/epoch-checkpoints/epoch-009.json; then echo complete; else echo incomplete; fi'"],capture_output=True,text=True,timeout=80)
  if x.returncode:raise RuntimeError('Snapshot observation lost; no publication intent or retry')
  if x.stdout.strip()=='complete':
   p=Path('artifacts/final-public-source-fit-evc40-20261013-v1/publication/epoch-009')
   if p.exists():raise RuntimeError('Existing publication attempt requires read-only reconciliation')
   x=subprocess.run([sys.executable,'artifacts/final-public-source-fit-evc40-20261013-v1/publication_coordinator.py','9'],capture_output=True,text=True,timeout=3900)
   (L/'publication-result.json').write_text(json.dumps({'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr,'automatic_retry':False},indent=2))
   if x.returncode:raise RuntimeError('Original publication did not verify; no retry')
   v=json.loads(x.stdout);assert v['epoch']==9 and v['file_count']==5
   (L/'verified.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'epoch':9,'revision':v['revision'],'status':v['status']}),flush=True);break
  time.sleep(45)
 else:raise TimeoutError('Finite observation ended before epoch9 snapshot; no publication intent')
except BaseException as e:
 (L/'terminal-failure.json').write_text(json.dumps({'error_type':type(e).__name__,'message':str(e),'automatic_retry':False},indent=2));raise
