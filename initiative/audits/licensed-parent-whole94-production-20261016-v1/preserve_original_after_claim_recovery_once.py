import hashlib,json,os,subprocess,sys,time
from pathlib import Path
L=Path(__file__).resolve().parent;intent=json.loads((L/'observation-recovered-v2.intent.json').read_text());owner=intent['PID'];start=time.monotonic();maximum=31*3600;files={n:hashlib.sha256((L/n).read_bytes()).hexdigest() for n in ('publish-program.py.txt','publish_epoch_after_reconnection_once.py')}
with(L/'observation-v2-preserve.intent.json').open('x') as f:json.dump({'PID':os.getpid(),'full_training_owner_PID':owner,'maximum_seconds':maximum,'maximum_publications':3,'immutable_files':files,'GPU':0,'automatic_retry':False},f,indent=2)
def progress(phase,**extra):(L/'observation-v2-preserve.progress.json').write_text(json.dumps({'phase':phase,'PID':os.getpid(),'elapsed_seconds':time.monotonic()-start,**extra},indent=2)+'\n')
try:
 results=[]
 for epoch in (1,2,3):
  receipt=L/('segment-'+str(epoch)+'-actual-accepted.json')
  while not receipt.exists():
   assert time.monotonic()-start<maximum
   if (L/'observation-recovered-v2.result.json').exists():raise RuntimeError('Original training controller stopped before this epoch receipt; no repeated publication/dispatch')
   os.kill(owner,0);command=subprocess.check_output(['ps','-p',str(owner),'-o','command='],text=True);assert 'continue_original_after_claim_contention_once.py' in command;progress('waiting_live_original_training_owner_and_actual_epoch',epoch=epoch);time.sleep(60)
  for n,h in files.items():assert hashlib.sha256((L/n).read_bytes()).hexdigest()==h
  progress('preserving_new_actual_immutable_epoch',epoch=epoch);x=subprocess.run([sys.executable,str(L/'publish_epoch_after_reconnection_once.py'),str(epoch)],capture_output=True,text=True,timeout=1600)
  with(L/('observation-v2-preserve-child-'+str(epoch)+'.json')).open('x') as f:json.dump({'epoch':epoch,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr},f,indent=2)
  assert x.returncode==0,'Original epoch preservation stopped; retain exact prefix/intent and reconcile read-only';outer=json.loads(x.stdout);assert outer['returncode']==0;value=json.loads(outer['stdout']);assert value['status']=='actual_whole94_parent_epoch_private_full_download_size_SHA_verified';results.append(value);progress('actual_epoch_private_full_verified',actual=value)
 result={'status':'all_three_original_whole94_parent_epochs_private_full_verified','actual':results};(L/'observation-v2-preserve.result.json').write_text(json.dumps(result,indent=2)+'\n');progress('terminal_success');print(json.dumps(result),flush=True)
except BaseException as e:
 result={'status':'stopped_original_parent_epoch_preservation_for_readonly_review','error_type':type(e).__name__,'reason':str(e),'automatic_retry':False};(L/'observation-v2-preserve.result.json').write_text(json.dumps(result,indent=2)+'\n');raise
