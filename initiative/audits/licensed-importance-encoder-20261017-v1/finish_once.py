"""Finite sole owner: one diagnostic push and exact read-only outcome, zero entries."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
L=Path(__file__).resolve().parent;ROOT=L.parents[1];files={n:hashlib.sha256((L/n).read_bytes()).hexdigest() for n in ('program.py.txt','run_phase_once.py')};assert (L/'staged.json').exists();start=time.monotonic();maximum=2*3600
with (L/'finish.intent.json').open('x') as f:json.dump({'PID':os.getpid(),'immutable_files':files,'maximum_seconds':maximum,'maximum_GPU_pushes':1,'competition_entries':0,'full_production_dispatches':0,'unfinished_primary_training_GPU_funding_preserved':True,'automatic_mutation_retry':False},f,indent=2)
def progress(phase,**extra):(L/'finish.progress.json').write_text(json.dumps({'phase':phase,'PID':os.getpid(),'elapsed_seconds':time.monotonic()-start,**extra},indent=2)+'\n')
def wait(seconds):
 until=time.monotonic()+seconds
 while time.monotonic()<until:
  assert time.monotonic()-start<maximum;time.sleep(min(60,until-time.monotonic()))
def run(phase):
 for n,h in files.items():assert hashlib.sha256((L/n).read_bytes()).hexdigest()==h
 x=subprocess.run([sys.executable,str(L/'run_phase_once.py'),phase],capture_output=True,text=True,env={**os.environ,'PYTHONPATH':str(ROOT),'PYTHONNOUSERSITE':'1'},timeout=1000)
 with (L/('child-'+str(len(list(L.glob('child-*.json'))))+'.json')).open('x') as f:json.dump({'phase':phase,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr},f,indent=2)
 assert x.returncode==0,'Original coordinator phase failed; preserve exact intent/result and reconcile read-only';outer=json.loads(x.stdout);assert outer['returncode']==0;return json.loads(outer['stdout'])
try:
 while True:
  progress('checking_authoritative_GPU_funding_before_one_diagnostic');value=run('push')
  if value['status']=='one_private_matched_IID16_real_MRI_recovery_pushed':break
  assert value['status']=='waiting_authoritative_GPU_budget_preserving_all_unfinished_original_epochs' and value['mutation_intent_created'] is False;progress('waiting_GPU_funding_without_new_intent',actual=value);wait(300)
 (L/'actual-pushed.json').write_text(json.dumps(value,indent=2)+'\n');progress('one_exact_private_diagnostic_pushed',actual=value);wait(90)
 while True:
  value=run('status');progress('observed_same_original_diagnostic',actual=value)
  if value['status'] in ('COMPLETE','ERROR','CANCELLED'):break
  wait(120)
 (L/'actual-terminal.json').write_text(json.dumps(value,indent=2)+'\n');result={'status':'matched_IID16_real_MRI_diagnostic_actual_terminal','actual':value,'competition_entries':0,'full_production_dispatches':0};(L/'finish.result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except BaseException as e:
 (L/'finish.result.json').write_text(json.dumps({'status':'original_diagnostic_owner_stopped_for_readonly_reconciliation','error_type':type(e).__name__,'reason':str(e),'automatic_retry':False},indent=2)+'\n');raise
