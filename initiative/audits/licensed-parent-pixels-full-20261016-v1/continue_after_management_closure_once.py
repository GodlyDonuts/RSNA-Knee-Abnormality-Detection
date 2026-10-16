import hashlib,json,os,subprocess,sys,time
from pathlib import Path
L=Path(__file__).resolve().parent;start=time.monotonic();files={n:hashlib.sha256((L/n).read_bytes()).hexdigest() for n in ('continuation-program.py.txt','run_continuation_phase_once.py')};maximum=10*3600
with (L/'continue-after-management-closure.intent.json').open('x') as f:json.dump({'PID':os.getpid(),'immutable_files':files,'maximum_seconds':maximum,'GPU_requested':False,'competition_entries':0,'maximum_new_CPU_notebooks':4,'automatic_mutation_retry':False},f,indent=2)
def progress(phase,**extra):(L/'continue-after-management-closure.progress.json').write_text(json.dumps({'phase':phase,'PID':os.getpid(),'elapsed_seconds':time.monotonic()-start,**extra},indent=2)+'\n')
def run(phase,shard=None):
 assert time.monotonic()-start<maximum
 for n,h in files.items():assert hashlib.sha256((L/n).read_bytes()).hexdigest()==h
 progress(phase,shard=shard);argv=[sys.executable,str(L/'run_continuation_phase_once.py'),phase]+([] if shard is None else [str(shard)]);x=subprocess.run(argv,capture_output=True,text=True,timeout=2750 if phase=='collect' else 700)
 with (L/('continued-child-'+str(len(list(L.glob('continued-child-*.json'))))+'.json')).open('x') as f:json.dump({'phase':phase,'shard':shard,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr},f,indent=2)
 assert x.returncode==0,'Original phase stopped; inspect original intents/results read-only, no automatic mutation retry';row=json.loads(x.stdout);assert row['returncode']==0;return json.loads(row['stdout'])
try:
 accepted=json.loads((L/'management-dependency-accepted.json').read_text());assert accepted['status']=='management_dependency_added_after_original_pre_API_import_failure' and accepted['all_four_remote_push_intents_and_ledger_rows_absent'] is True;staged=json.loads((L/'staged.json').read_text());pushed=[]
 for i in range(4):pushed.append(run('push',i))
 progress('four_original_CPU_Kaggle_shards_dispatched',pushed=pushed);complete={};verified={};next_poll=time.monotonic()
 while len(verified)<4:
  assert time.monotonic()-start<maximum
  if time.monotonic()<next_poll:time.sleep(min(60,next_poll-time.monotonic()));continue
  for i in range(4):
   if i in verified:continue
   actual=run('status',i)
   if actual['status'] in ('ERROR','CANCELLED'):raise RuntimeError('Original CPU cache shard failed; preserve original and inspect exact trace')
   if actual['status']=='COMPLETE':
    assert 'receipt' in actual;complete[i]=actual;verified[i]=run('collect',i)
  progress('original_CPU_cache_work_observed',pushed=pushed,complete_shards=sorted(complete),verified_shards=sorted(verified));next_poll=time.monotonic()+300
 result={'status':'four_original_Kaggle_pixel_shards_and_complete3887_inputs_actual_verified','pushed':pushed,'actual_Kaggle':list(complete.values()),'actual_collected':list(verified.values()),'GPU_used':False,'competition_entries':0};(L/'continue-after-management-closure.result.json').write_text(json.dumps(result,indent=2)+'\n');progress('terminal_success');print(json.dumps(result),flush=True)
except BaseException as e:
 (L/'continue-after-management-closure.result.json').write_text(json.dumps({'status':'stopped_original_full_pixel_cache_requires_readonly_review','error_type':type(e).__name__,'reason':str(e),'automatic_retry':False},indent=2)+'\n');raise
