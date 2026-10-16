import hashlib,json,os,subprocess,sys,time
from pathlib import Path
L=Path(__file__).resolve().parent;stopped=json.loads((L/'concurrent-collection-observer.result.json').read_text());assert stopped['status']=='stopped_original_collection_observer_for_readonly_review';failed=json.loads((L/'concurrent-observer-collect-0-result-0.json').read_text());assert failed['returncode']==1 and 'Original stream failed' in failed['stderr'];start=time.monotonic();maximum=10*3600;files={n:hashlib.sha256((L/n).read_bytes()).hexdigest() for n in ('collection-bounded-program.py.txt','run_bounded_collection_phase_once.py')}
with(L/'bounded-collection-observer.intent.json').open('x') as f:json.dump({'PID':os.getpid(),'immutable_files':files,'maximum_seconds':maximum,'Kaggle_pushes':0,'competition_entries':0,'GPU':0,'original_four_sources_and_versions_unchanged':True,'automatic_mutation_retry':False},f,indent=2)
def progress(phase,**extra):(L/'bounded-collection-observer.progress.json').write_text(json.dumps({'phase':phase,'PID':os.getpid(),'elapsed_seconds':time.monotonic()-start,**extra},indent=2)+'\n')
def run(phase,shard=None):
 for n,h in files.items():assert hashlib.sha256((L/n).read_bytes()).hexdigest()==h
 progress(phase,shard=shard);x=subprocess.run([sys.executable,str(L/'run_bounded_collection_phase_once.py'),phase]+([] if shard is None else [str(shard)]),capture_output=True,text=True,timeout=2750 if phase=='collect' else 700)
 with(L/('bounded-observer-child-'+str(len(list(L.glob('bounded-observer-child-*.json'))))+'.json')).open('x') as f:json.dump({'phase':phase,'shard':shard,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr},f,indent=2)
 assert x.returncode==0,'Original observation/collection stopped; preserve original receipt/prefix and reconcile read-only';result=json.loads(x.stdout);assert result['returncode']==0;return json.loads(result['stdout'])
try:
 verified={};complete={};next_read=time.monotonic()
 while len(verified)<4:
  assert time.monotonic()-start<maximum
  if time.monotonic()<next_read:time.sleep(min(60,next_read-time.monotonic()));continue
  actual=run('status_all');assert len(actual)==4
  for i,row in enumerate(actual):
   assert row['shard']==i
   if row['status'] in ('ERROR','CANCELLED'):raise RuntimeError('Actual original CPU pixel shard failed; preserve exact source and inspect original trace')
   if row['status']=='COMPLETE' and i not in verified:assert 'receipt' in row;complete[i]=row;verified[i]=run('collect',i)
  progress('all_original_four_observed_in_one_CPU_allocation',actual=actual,verified_shards=sorted(verified));next_read=time.monotonic()+300
 result={'status':'all_original_four_CPU_pixel_shards_complete_and_collected_verified','actual_Kaggle':list(complete.values()),'actual_collected':list(verified.values()),'GPU_used':False,'competition_entries':0,'new_notebook_pushes':0};(L/'bounded-collection-observer.result.json').write_text(json.dumps(result,indent=2)+'\n');progress('terminal_success');print(json.dumps(result),flush=True)
except BaseException as e:
 (L/'bounded-collection-observer.result.json').write_text(json.dumps({'status':'stopped_original_collection_observer_for_readonly_review','error_type':type(e).__name__,'reason':str(e),'automatic_retry':False},indent=2)+'\n');raise
