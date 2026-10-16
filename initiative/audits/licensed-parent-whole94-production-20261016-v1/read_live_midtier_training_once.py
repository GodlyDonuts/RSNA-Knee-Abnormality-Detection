import json,shlex,subprocess
from pathlib import Path
L=Path(__file__).resolve().parent;actual=json.loads((L/'segment-1-pushed.json').read_text());program=r'''
import json,os,sqlite3,sys,time
import requests
v=json.load(sys.stdin);assert os.uname().nodename=='evc33' and not os.environ.get('SLURM_JOB_GPUS');actual=v['actual']
with sqlite3.connect('file:/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3?mode=ro',uri=True) as db:record=json.loads(db.execute('SELECT value FROM meta WHERE key=?',('licensed-parent-whole94-full-fit:'+actual['source_sha256'],)).fetchone()[0])
assert record['kernel_id']==actual['kernel_id'] and record['slug']==actual['slug'] and record['version']==1
start=time.monotonic();samples=[];seen=0;http=None;content=None;error=None;s=requests.Session();s.headers.update({'Authorization':'Bearer '+v['token'],'Accept':'text/event-stream, */*'})
try:
 response=s.get('https://www.kaggle.com/api/v1/kernels/logs/stream/'+actual['slug'],params={'versionLabel':'v1'},stream=True,timeout=(10,15),allow_redirects=False);http=response.status_code;content=response.headers.get('Content-Type');assert http==200
 try:
  for raw in response.iter_lines(chunk_size=1,decode_unicode=True):
   if time.monotonic()-start>45 or seen>2**20:break
   if not raw:continue
   if isinstance(raw,bytes):raw=raw.decode('utf-8','replace')
   seen+=len(raw)
   if raw.startswith('data:'):
    text=raw[5:].lstrip()
    if text=='END_OF_LOG':break
    try:event=json.loads(text);data=event.get('data','')
    except (ValueError,AttributeError):continue
   else:data=raw
   if not isinstance(data,str):continue
   for line in data.splitlines():
    try:value=json.loads(line)
    except ValueError:continue
    if isinstance(value,dict) and {'segment','allocated_headers','actual_optimizer_progress','elapsed_seconds'}<=value.keys():samples.append({k:value[k] for k in ('segment','allocated_headers','actual_optimizer_progress','elapsed_seconds')});samples=samples[-3:]
 finally:response.close()
except requests.RequestException as exc:error=type(exc).__name__
result={'status':'actual_original_live_Kaggle_training_stream_readonly','CPU_job':os.environ['SLURM_JOB_ID'],'kernel_id':actual['kernel_id'],'source_sha256':actual['source_sha256'],'version':1,'HTTP_status':http,'content_type':content,'read_bytes':seen,'aggregate_training_samples':samples,'read_error_type':error,'elapsed_seconds':time.monotonic()-start,'new_GPU_job_or_mutation':False};print(json.dumps(result),flush=True)
'''
E=Path('.env');assert E.stat().st_mode&0o077==0;values={k.strip():val.strip().strip('"\'') for line in E.read_text().splitlines() for k,sep,val in [line.partition('=')] if sep};cmd=['srun','--account=skattel','--partition=normal','--nodelist=evc33','--ntasks=1','--cpus-per-task=1','--gres=none','--mem=1G','--time=00:02:00','--job-name=rsna-parent-live-optimizer-read','/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python','-B','-c',program];x=subprocess.run(['ssh','-S','/tmp/rsna-research-20261001-stable.sock','-o','BatchMode=yes','newton',shlex.join(cmd)],input=json.dumps({'actual':actual,'token':values['KAGGLE_API_TOKEN']}),capture_output=True,text=True,timeout=150);out={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr};name='live-midtier-stream-readonly-'+str(len(list(L.glob('live-stream-readonly-*.json'))))+'.json';(L/name).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
