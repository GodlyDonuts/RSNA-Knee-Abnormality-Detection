import hashlib
import json
import shlex
import subprocess
from pathlib import Path

L = Path(__file__).resolve().parent
ROOT = L.parents[1]
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
names = ['rsna/competitive/licensed_pixel_cache.py', 'rsna/competitive/licensed_pixel_bags.py']
files = {n: subprocess.check_output(['git', 'show', commit + ':' + n], cwd=ROOT, text=True) for n in names}
program = r'''
import ast,base64,hashlib,io,json,os,shutil,sys,time
from pathlib import Path
import numpy as np,torch
v=json.load(sys.stdin);assert os.uname().nodename=='evc33' and not os.environ.get('SLURM_JOB_GPUS');R=Path('/tmp/rsna-licensed-parent-production-inputs-20261016-v1');O=R/'actual32-bridge';assert not O.exists();assert shutil.disk_usage('/tmp').free>607095306179
controls=[R,Path('/tmp/rsna-licensed-parent-trainer-recovery-20261016-v1'),Path('/tmp/rsna-licensed-parent-whole94-recovery-20261016-v1')];used=sum(p.stat().st_size for root in controls for p in root.rglob('*') if p.is_file());assert used+2**20<16*2**20;O.mkdir(mode=0o700);code=O/'code'
for name,text in {**v['files'],'rsna/__init__.py':'','rsna/competitive/__init__.py':''}.items():
 p=code/name;assert p.resolve().is_relative_to(code.resolve());p.parent.mkdir(mode=0o700,parents=True,exist_ok=True);p.write_text(text);p.chmod(0o600)
sys.path.insert(0,str(code));from rsna.competitive.licensed_pixel_bags import PixelBags
torch.set_num_threads(1);torch.set_num_interop_threads(1);P=Path('/tmp/rsna-licensed-parent-pixels32-20261016-v1');D=P/'collected';prior=json.loads((D/'verified.json').read_text());assert prior['CPU_job']=='938918' and prior['all32_original_NPZ_volume_mask_bytes_and_rendered_two_view_hashes_exact'];index_raw=(D/'index.private.json').read_bytes();assert hashlib.sha256(index_raw).hexdigest()=='3a1567989bf65779aeb19785dafdedef7386fb6bedba5ec45da6d3a67a9a79e3';rows=json.loads(index_raw)['records'];assert len(rows)==32
source=(P/'pixel-cache.private.py').read_text();assert hashlib.sha256(source.encode()).hexdigest()=='09254e97913d8dbab543825660df114242d9314315ebb8d4550ef7c7ceb388c9';ns={'__name__':'actual_original_CPU_renderer','__file__':str(P/'pixel-cache.private.py')};exec(compile(source,'unchanged_original_CPU_renderer','exec'),ns)
bootstrap=Path('/tmp/rsna-licensed-parent-whole94-recovery-20261016-v1/recovery.private.py').read_text();assert hashlib.sha256(bootstrap.encode()).hexdigest()=='a45d2a00322d84e412c4bd28235fc49d81a42f3ec3e733f9b7da76f68ec57b39';nodes=[n for n in ast.parse(bootstrap).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TARGETS' for t in n.targets)];assert len(nodes)==1;raw=base64.b64decode(ast.literal_eval(nodes[0].value));assert hashlib.sha256(raw).hexdigest()=='87cbd096202eb3a0512f918b181ecd765c893817beed4678e4eea2b47032c834';old=json.loads(raw);uids=[r['study_uid'] for r in rows];assert old['uids']==uids and old['eligible_fitting_only']
target=R/'targets.private.npz';assert target.stat().st_size==483656 and hashlib.sha256(target.read_bytes()).hexdigest()=='48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285'
with np.load(target,allow_pickle=False) as arrays:full={k:arrays[k].copy() for k in arrays.files}
assert len(full['uids'])==3887 and hashlib.sha256(json.dumps(full['uids'].tolist(),separators=(',',':')).encode()).hexdigest()=='0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42';lookup={uid:i for i,uid in enumerate(full['uids'].tolist())};positions=[lookup[u] for u in uids];targets={k:array[positions] for k,array in full.items()};assert np.array_equal(targets['labels'],np.asarray(old['labels'],dtype=np.float32),equal_nan=True) and np.array_equal(targets['masks'],np.asarray(old['masks'],dtype=np.bool_))
F=Path('/tmp/rsna-released-context-head-fit-20261016-v1/features32');feature_raw=(F/'index.private.json').read_bytes();assert len(feature_raw)==8574 and hashlib.sha256(feature_raw).hexdigest()=='a1bba9e618444f16ddd453efa1bb8650245f8259dcd9a00348b17efd87c1e100';features=json.loads(feature_raw)['records'];assert [r['study_uid'] for r in features]==uids
for i,row in enumerate(features):
 raw=(F/row['file']).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 with np.load(io.BytesIO(raw),allow_pickle=False) as arrays:assert np.array_equal(arrays['parent_logits'],targets['parent_logits'][i])
records=[{**r,'holder':'probe32'} for r in rows];render=lambda volume,mask,mirror:ns['eval_windows'](volume,mask,94,320,norm=ns['NORM'],mirror=mirror);data=PixelBags(records,{'probe32':D/'pixels.tar'},targets,render,diagnostic_only=True);start=time.monotonic()
for i in range(32):
 for mirror in (False,True):
  case=data.case(i,list(range(94)),mirror);row=rows[i];assert hashlib.sha256(case['images'].numpy().tobytes()).hexdigest()==row['original_mirrored_input_sha256' if mirror else 'original_plain_input_sha256'];assert torch.equal(case['labels'],torch.from_numpy(targets['labels'][i])) and torch.equal(case['mask'],torch.from_numpy(targets['masks'][i])) and torch.equal(case['parent_logits'],torch.from_numpy(targets['parent_logits'][i,int(mirror)]))
result={'status':'actual32_full_production_PixelBags_CPU_bridge_bit_exact_accepted','CPU_job':os.environ['SLURM_JOB_ID'],'source_git':v['source_git'],'source_modules':{n:hashlib.sha256(s.encode()).hexdigest() for n,s in v['files'].items()},'original_pixel_receipt_CPU':'938918','original_whole94_GPU':139105348,'target_dataset_id':12641441,'target_dataset_version':1,'target_sha256':'48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285','studies':32,'views_per_study':2,'windows_per_view':94,'exact_original_input_hashes':True,'original_GPU_diagnostic_source_labels_and_masks_bit_equal':True,'both_original_feature_teacher_views_bit_equal':True,'direct_tar_NPZ_read_no_second_pixel_tree':True,'elapsed_seconds':time.monotonic()-start,'GPU_or_encoder_or_optimizer_or_clinical_evaluation_or_network_used':False,'new_Kaggle_mutation':False,'full3887_cache_admitted':False};(O/'verified.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');assert sum(p.stat().st_size for root in controls for p in root.rglob('*') if p.is_file())<16*2**20;print(json.dumps(result),flush=True)
'''
compile(program, 'actual32_production_bridge', 'exec')
(L / 'actual32-bridge-program.py.txt').write_text(program)
with (L / 'actual32-bridge.intent.json').open('x') as f:
    json.dump({'source_git':commit,'program_sha256':hashlib.sha256(program.encode()).hexdigest(),'new_MRI_downloads_or_reencoding':False,'CPU':1,'GPU':0,'automatic_retry':False}, f, indent=2)
cmd=['srun','--account=skattel','--partition=normal','--nodes=1','--nodelist=evc33','--ntasks=1','--cpus-per-task=1','--gres=none','--mem=4G','--time=00:05:00','--job-name=rsna-parent-full-inputs32','env','PYTHONDONTWRITEBYTECODE=1','PYTHONNOUSERSITE=1','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python','-B','-c',program]
x=subprocess.run(['ssh','-S','/tmp/rsna-research-20261001-stable.sock','-o','BatchMode=yes','newton',shlex.join(cmd)],input=json.dumps({'source_git':commit,'files':files}),capture_output=True,text=True,timeout=350)
result={'returncode':x.returncode,'stdout':x.stdout,'stderr':x.stderr}
(L/'actual32-bridge.result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
if x.returncode:raise SystemExit(1)
(L/'actual32-bridge-accepted.json').write_text(json.dumps(json.loads(x.stdout),indent=2)+'\n')
