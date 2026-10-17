"""Distinct finite continuation: reconcile existing push before new epoch dispatch."""
import hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
L=Path(__file__).resolve().parent;ROOT=L.parents[1];start=time.monotonic();maximum=30*3600
old=json.loads((L/'finish.intent.json').read_text());assert (L/'finish.result.json').exists()
try:os.kill(old['PID'],0)
except ProcessLookupError:pass
else:raise RuntimeError('Original controller still live')
program=(L/'program.py.txt').read_text();program_sha=hashlib.sha256(program.encode()).hexdigest();assert program_sha==old['immutable_files']['program.py.txt'];meta=json.loads((L/'staged.json').read_text());intent=L/'observation-recovered-v2.intent.json'
with intent.open('x') as f:json.dump({'PID':os.getpid(),'previous_owner':old['PID'],'program_sha256':program_sha,'maximum_seconds':maximum,'new_full_fit_identities':False,'resume_original_existing_segments_only':True,'automatic_mutation_retry':False},f,indent=2)
def progress(phase,**extra):(L/'observation-recovered-v2.progress.json').write_text(json.dumps({'phase':phase,'PID':os.getpid(),'elapsed_seconds':time.monotonic()-start,**extra},indent=2)+'\n')
def phase(name,segment):
 assert hashlib.sha256((L/'program.py.txt').read_bytes()).hexdigest()==program_sha
 E=ROOT/'.env';assert E.stat().st_mode&0o077==0;values={k.strip():v.strip().strip('"\'') for line in E.read_text().splitlines() for k,sep,v in [line.partition('=')] if sep};v={'phase':name,'segment':segment,'program':program,'source_git':meta['source_git'],'metadata':meta,'token':values['KAGGLE_API_TOKEN']};bootstrap='import json,sys;v=json.load(sys.stdin);exec(compile(v["program"],"original_whole94_phase","exec"),{"__name__":"__main__","PACKAGE":v})';cmd=['srun','--account=skattel','--partition=preemptable','--qos=preemptable','--nodelist=evc33','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=none','--mem=4G','--time=00:15:00','--job-name=rsna-parent-reconciled-'+name,'env','PYTHONDONTWRITEBYTECODE=1','PYTHONNOUSERSITE=1','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python','-B','-c',bootstrap]
 
 for attempt in range(8):
  x=subprocess.run(['ssh','-S','/tmp/rsna-research-20261001-stable.sock','-o','BatchMode=yes','newton',shlex.join(cmd)],input=json.dumps(v),capture_output=True,text=True,timeout=950)
  if x.returncode and name=='status' and 'Canonical coordinator remains busy' in x.stderr:
   progress('readonly_status_waiting_canonical_claim_without_mutation',segment=segment,attempt=attempt);wait(60);continue
  break
 raw={'phase':name,'segment':segment,'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr};path=L/('observation-recovered-v2-child-'+str(len(list(L.glob('observation-recovered-v2-child-*.json'))))+'.json');path.write_text(json.dumps(raw,indent=2)+'\n');assert x.returncode==0,'Preserve original uncertain mutation or observation for read-only review';return json.loads(x.stdout)
def wait(seconds):
 until=time.monotonic()+seconds
 while time.monotonic()<until:assert time.monotonic()-start<maximum;time.sleep(min(60,until-time.monotonic()))
try:
 completed=[]
 for segment in (1,2,3):
  local=L/('segment-'+str(segment)+'-pushed.json')
  if local.exists():pushed=json.loads(local.read_text());assert pushed['segment']==segment
  else:
   assert segment>1 and completed[-1]['status']=='COMPLETE'
   progress('preparing_only_next_original_unpushed_epoch',segment=segment);ready=phase('prepare',segment);assert ready['status']=='one_whole94_full_session_prepared'
   pushed=phase('push',segment);assert pushed['status']=='one_whole94_full_session_pushed';local.write_text(json.dumps(pushed,indent=2)+'\n')
  while True:
   actual=phase('status',segment);progress('reconciled_original_actual_state',actual=actual)
   assert actual['status'] not in ('ERROR','CANCELLED'),'Original actual training failed: no new epoch or retry'
   if actual['status']=='COMPLETE':
    assert actual['receipt']['completed']==486*segment;completed.append(actual);p=L/('segment-'+str(segment)+'-actual-accepted.json')
    if p.exists():assert json.loads(p.read_text())==actual
    else:p.write_text(json.dumps(actual,indent=2)+'\n')
    break
   wait(300)
 result={'status':'all_original_three_whole94_sessions_complete_after_observation_reconciliation','actual':completed,'no_duplicate_push':True};(L/'observation-recovered-v2.result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
except BaseException as e:
 (L/'observation-recovered-v2.result.json').write_text(json.dumps({'status':'stopped_for_exact_readonly_review','error':type(e).__name__,'reason':str(e),'automatic_retry':False},indent=2)+'\n');raise
