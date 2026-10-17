import hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
L=Path(__file__).resolve().parent;source_owner=json.loads((L/'observation-v2-serving.intent.json').read_text())['PID'];files={n:hashlib.sha256((L/n).read_bytes()).hexdigest() for n in ('entry-program.py.txt','run_entry_after_reconnection_once.py')};entry_deadline=datetime(2026,10,17,23,10,tzinfo=timezone.utc).timestamp();outcome_deadline=datetime(2026,10,18,10,0,tzinfo=timezone.utc).timestamp();start=time.monotonic()
with (L/'entry-observation-v2-finish.intent.json').open('x') as f:json.dump({'PID':os.getpid(),'source_serving_owner_PID':source_owner,'immutable_files':files,'maximum_code_entries':1,'exact_epoch':1,'entry_deadline_UTC':datetime.fromtimestamp(entry_deadline,timezone.utc).isoformat(),'outcome_deadline_UTC':datetime.fromtimestamp(outcome_deadline,timezone.utc).isoformat(),'actual_prior_scored_outcome_and_no_unresolved_entries_required':True,'no_public_CSV_upload':True,'automatic_retry':False},f,indent=2)
def progress(phase,**extra):(L/'entry-observation-v2-finish.progress.json').write_text(json.dumps({'phase':phase,'PID':os.getpid(),'elapsed_seconds':time.monotonic()-start,**extra},indent=2)+'\n')
def wait(seconds,deadline):
 until=min(time.time()+seconds,deadline)
 while time.time()<until:time.sleep(min(60,max(.01,until-time.time())))
def run(phase):
 for n,h in files.items():assert hashlib.sha256((L/n).read_bytes()).hexdigest()==h
 progress(phase);x=subprocess.run([sys.executable,str(L/'run_entry_after_reconnection_once.py'),phase],capture_output=True,text=True,timeout=1000)
 with (L/('entry-observation-v2-child-'+str(len(list(L.glob('entry-observation-v2-child-*.json'))))+'.json')).open('x') as f:json.dump({'phase':phase,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr},f,indent=2)
 assert x.returncode==0,'Original single-epoch entry coordinator stopped; retain intent and reconcile read-only';outer=json.loads(x.stdout);assert outer['returncode']==0;return json.loads(outer['stdout'])
try:
 submitted=None
 while time.time()<entry_deadline:
  if not (L/'epoch-1-competition-actual-accepted.json').exists():
   if (L/'observation-v2-serving.result.json').exists():raise RuntimeError('Original serving controller stopped before epoch1 public acceptance')
   os.kill(source_owner,0);assert 'qualify_original_after_claim_recovery_once.py' in subprocess.check_output(['ps','-p',str(source_owner),'-o','command='],text=True);progress('waiting_live_original_serving_and_actual_epoch1_qualification');wait(60,entry_deadline);continue
  value=run('eligibility')
  if value['eligible'] is True:submitted=run('submit');assert submitted['status']=='one_exact_whole94_full_parent_epoch1_code_submission_accepted';(L/'epoch-1-actual-submitted.json').write_text(json.dumps(submitted,indent=2)+'\n');progress('one_exact_epoch1_code_entry_accepted_waiting_actual_hidden_outcome',submission=submitted);break
  progress('waiting_actual_prior_outcomes_and_fresh_allowance',actual=value);wait(600,entry_deadline)
 if submitted is None:result={'status':'finite_epoch1_entry_window_expired_without_submission','competition_entries':0,'automatic_retry':False}
 else:
  result=None;wait(180,outcome_deadline)
  while time.time()<outcome_deadline:
   value=run('outcome');progress('actual_epoch1_hidden_outcome_observed',actual=value)
   if value['outcome'] in ('scored','failed'):result={'status':'one_exact_epoch1_submission_actual_hidden_terminal','actual':value,'competition_entries':1};break
   wait(300,outcome_deadline)
  if result is None:result={'status':'finite_epoch1_hidden_observation_expired_pending_or_unknown','submission':submitted,'competition_entries':1,'automatic_retry':False}
 (L/'entry-observation-v2-finish.result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except BaseException as e:
 result={'status':'stopped_original_single_epoch1_entry_for_readonly_review','error_type':type(e).__name__,'reason':str(e),'automatic_retry':False};(L/'entry-observation-v2-finish.result.json').write_text(json.dumps(result,indent=2)+'\n');raise
