import hashlib,json,os,subprocess,sys,time
from pathlib import Path
L=Path(__file__).resolve().parent;start=time.monotonic();maximum=30*3600;files={n:hashlib.sha256((L/n).read_bytes()).hexdigest() for n in ('program.py.txt','run_phase_once.py')};assert (L/'staged.json').exists()
with (L/'finish.intent.json').open('x') as f:json.dump({'PID':os.getpid(),'immutable_files':files,'maximum_seconds':maximum,'maximum_GPU_notebook_pushes':3,'new_competition_entries':0,'actual_original_full_pixel_receipts_before_GPU':True,'whole94_fixed3_epochs_fixed1458_schedule':True,'automatic_mutation_retry':False},f,indent=2)
def progress(phase,**extra):(L/'finish.progress.json').write_text(json.dumps({'phase':phase,'PID':os.getpid(),'elapsed_seconds':time.monotonic()-start,**extra},indent=2)+'\n')
def run(phase,segment):
 for n,h in files.items():assert hashlib.sha256((L/n).read_bytes()).hexdigest()==h
 progress(phase,segment=segment);x=subprocess.run([sys.executable,str(L/'run_phase_once.py'),phase,str(segment)],capture_output=True,text=True,timeout=1000)
 with(L/('child-'+str(len(list(L.glob('child-*.json'))))+'.json')).open('x') as f:json.dump({'phase':phase,'segment':segment,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr},f,indent=2)
 assert x.returncode==0,'Original full-training coordinator stopped; preserve exact intent/state and reconcile read-only';outer=json.loads(x.stdout);assert outer['returncode']==0;return json.loads(outer['stdout'])
def wait_until(at):
 while time.monotonic()<at:assert time.monotonic()-start<maximum;time.sleep(min(60,max(.01,at-time.monotonic())))
try:
 completed=[]
 for segment in (1,2,3):
  while True:
   assert time.monotonic()-start<maximum;ready=run('prepare',segment)
   if ready['status']=='one_whole94_full_session_prepared':break
   assert segment==1 and ready['status']=='waiting_for_exact_original_pixel_producers';progress('waiting_actual_original_full_pixel_producers',actual=ready['actual'],GPU_reserved_or_dispatched=False);wait_until(time.monotonic()+300)
  pushed=run('push',segment);assert pushed['status']=='one_whole94_full_session_pushed' and pushed['segment']==segment
  (L/('segment-'+str(segment)+'-pushed.json')).write_text(json.dumps(pushed,indent=2)+'\n');progress('original_full_fit_session_pushed',actual=pushed);wait_until(time.monotonic()+120)
  while True:
   actual=run('status',segment);assert actual['status'] not in ('ERROR','CANCELLED'),'Actual original whole94 full-training failure; do not dispatch the next session'
   if actual['status']=='COMPLETE':assert 'receipt' in actual;completed.append(actual);(L/('segment-'+str(segment)+'-actual-accepted.json')).write_text(json.dumps(actual,indent=2)+'\n');progress('actual_full_epoch_completed_checkpoint_preserved_in_original_private_Kaggle_output',actual=actual);break
   progress('observed_original_full_training_session',actual=actual);wait_until(time.monotonic()+300)
 result={'status':'all_three_checkpoint_continuous_whole94_full_parent_sessions_actual_complete','actual':completed,'new_competition_entries':0,'private_HF_epoch_publication_still_separate':True,'clinical_or_competition_accuracy_claim':False};(L/'finish.result.json').write_text(json.dumps(result,indent=2)+'\n');progress('terminal_success');print(json.dumps(result),flush=True)
except BaseException as e:
 result={'status':'stopped_original_full_whole94_controller_for_readonly_review','error_type':type(e).__name__,'reason':str(e),'automatic_retry':False};(L/'finish.result.json').write_text(json.dumps(result,indent=2)+'\n');raise
