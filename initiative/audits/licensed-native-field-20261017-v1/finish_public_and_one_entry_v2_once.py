"""Finite original384 public acceptance then one useful exact code entry; never repeat mutations."""
import hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
L=Path(__file__).resolve().parent;files={n:hashlib.sha256((L/n).read_bytes()).hexdigest() for n in ('cloud-public-v2.py.txt','run_public_v2_phase_once.py','competition.private.py')};accepted=json.loads((L/'actual-T4-qualified.json').read_text());assert accepted['status']=='COMPLETE' and accepted['kernel_id']==139126353;entry_deadline=datetime.fromisoformat('2026-10-17T23:25:00+00:00').timestamp();outcome_deadline=datetime.fromisoformat('2026-10-18T10:00:00+00:00').timestamp();start=time.monotonic()
with (L/'finish-public-entry-v2.intent.json').open('x') as f:json.dump({'PID':os.getpid(),'immutable_files':files,'maximum_new_public_GPU_notebooks':1,'maximum_new_code_submissions':1,'new_Dataset_uploads':0,'actual_qualification_kernel':139126353,'entry_deadline_UTC':'2026-10-17T23:25:00+00:00','outcome_deadline_UTC':'2026-10-18T10:00:00+00:00','fund_all_unfinished_primary_epochs_and_live_optional_runs':True,'fresh_account_allowance_and_actual_prior_scored_outcome_required':True,'public_CSV_upload':False,'automatic_mutation_retry':False},f,indent=2)
def progress(phase,**extra):(L/'finish-public-entry-v2.progress.json').write_text(json.dumps({'phase':phase,'PID':os.getpid(),'UTC':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,**extra},indent=2)+'\n')
def wait(seconds,deadline):
 end=min(deadline,time.time()+seconds)
 while time.time()<end:time.sleep(min(60,end-time.time()))
 assert time.time()<deadline,'Finite original observation deadline; reconcile actual state'
def run(phase):
 for n,h in files.items():assert hashlib.sha256((L/n).read_bytes()).hexdigest()==h
 x=subprocess.run([sys.executable,str(L/'run_public_v2_phase_once.py'),phase],capture_output=True,text=True,env={**os.environ,'PYTHONPATH':str(L.parents[1]),'PYTHONNOUSERSITE':'1'},timeout=700);child=L/('public-entry-v2-child-'+str(len(list(L.glob('public-entry-child-*.json'))))+'.json');child.write_text(json.dumps({'phase':phase,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr},indent=2)+'\n');assert x.returncode==0,'Original phase failed; reconcile its exact intent/result, no mutation retry';courier=json.loads(x.stdout);assert courier['returncode']==0;return json.loads(courier['stdout'])
try:
 while time.time()<entry_deadline:
  progress('checking_actual_VM_funding_before_one_public_push');value=run('push')
  if value['status']=='one_qualified_full384_public_MRI_notebook_pushed':break
  assert value['status']=='waiting_actual_VM_allowance_preserving_all_unfinished_primary_epochs_and_live_optional_runs' and value['mutation_intent_created'] is False;progress('waiting_actual_primary_and_optional_VM_funding',actual=value);wait(600,entry_deadline)
 else:raise TimeoutError('No funded public push in finite window')
 (L/'actual-public-pushed.json').write_text(json.dumps(value,indent=2)+'\n');progress('one_public_MRI_pushed',actual=value);wait(90,entry_deadline)
 while True:
  value=run('status');assert value['status'] not in ('ERROR','CANCELLED'),'Actual publicMRI failed; never repush'
  if value['status']=='COMPLETE':assert value['receipt']['input_side']==384 and 'CSV_verified' in value;break
  progress('actual_public_MRI_running',actual=value);wait(120,entry_deadline)
 (L/'actual-public-accepted.json').write_text(json.dumps(value,indent=2)+'\n');progress('actual_public_MRI_and_CSV_accepted',actual=value)
 while True:
  value=run('eligibility');progress('fresh_actual_entry_eligibility',actual=value)
  if value['eligible']:break
  wait(300,entry_deadline)
 value=run('submit');assert value['status']=='one_exact_unchanged_parent_field384_code_submission_accepted';(L/'actual-submitted.json').write_text(json.dumps(value,indent=2)+'\n');progress('one_exact_field384_entry_accepted',actual=value);wait(180,outcome_deadline)
 while True:
  actual=run('outcome');progress('actual_existing_field384_hidden_outcome',actual=actual)
  if actual['outcome'] in ('scored','failed'):break
  wait(300,outcome_deadline)
 result={'status':'one_original_field384_entry_actual_hidden_terminal','actual':actual,'maximum_competition_entries':1,'automatic_retry':False};(L/'finish-public-entry-v2.result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except BaseException as e:
 result={'status':'original_field384_public_or_entry_stopped_for_readonly_reconciliation','error_type':type(e).__name__,'reason':str(e),'automatic_retry':False};(L/'finish-public-entry-v2.result.json').write_text(json.dumps(result,indent=2)+'\n');raise
