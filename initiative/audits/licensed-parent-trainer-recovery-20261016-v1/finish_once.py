import hashlib,json,os,subprocess,sys,time
from pathlib import Path
L=Path(__file__).resolve().parent;start=time.monotonic();files={n:hashlib.sha256((L/n).read_bytes()).hexdigest() for n in ('program.py.txt','run_phase_once.py')};maximum=5400
with (L/'finish.intent.json').open('x') as f:json.dump({'PID':os.getpid(),'immutable_files':files,'maximum_seconds':maximum,'maximum_GPU_notebooks':1,'competition_entries':0,'automatic_mutation_retry':False},f,indent=2)
def progress(phase,**extra):(L/'finish.progress.json').write_text(json.dumps({'phase':phase,'PID':os.getpid(),'elapsed_seconds':time.monotonic()-start,**extra},indent=2)+'\n')
def run(phase):
 assert time.monotonic()-start<maximum
 for n,h in files.items():assert hashlib.sha256((L/n).read_bytes()).hexdigest()==h
 progress(phase);x=subprocess.run([sys.executable,str(L/'run_phase_once.py'),phase],capture_output=True,text=True,timeout=700)
 with (L/('child-'+str(len(list(L.glob('child-*.json'))))+'.json')).open('x') as f:json.dump({'phase':phase,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr},f,indent=2)
 assert x.returncode==0,'Original phase failed; reconcile intents/results read-only, never repeat mutation';row=json.loads(x.stdout);assert row['returncode']==0;return json.loads(row['stdout'])
try:
 staged=run('stage');pushed=run('push');assert pushed['status']=='one_exact_private_real_MRI_twoT4_recovery_pushed';progress('await_actual_real_MRI_CUDA_acceptance',pushed=pushed);next_read=time.monotonic()+90
 while time.monotonic()-start<maximum:
  if time.monotonic()<next_read:time.sleep(min(60,next_read-time.monotonic()));continue
  actual=run('status');progress('actual_CUDA_recovery_observed',actual=actual)
  if actual['status']=='COMPLETE':assert 'receipt' in actual;break
  if actual['status'] in ('ERROR','CANCELLED'):raise RuntimeError('Actual GPU recovery failed; preserve original trace/model/input before a correction')
  next_read=time.monotonic()+180
 else:raise RuntimeError('Finite observation expired; inspect original notebook read-only')
 result={'status':'original_real_MRI_parent_twoT4_recovery_actual_accepted','pushed':pushed,'actual':actual,'full_fit_dispatched':False,'competition_entries':0};(L/'finish.result.json').write_text(json.dumps(result,indent=2)+'\n');progress('terminal_success');print(json.dumps(result),flush=True)
except BaseException as e:
 (L/'finish.result.json').write_text(json.dumps({'status':'stopped_original_real_MRI_recovery_requires_readonly_review','error_type':type(e).__name__,'reason':str(e),'automatic_retry':False},indent=2)+'\n');raise
