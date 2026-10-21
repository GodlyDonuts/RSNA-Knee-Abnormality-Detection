"""Bounded single-T4 Colab measurements of actual fixed-final B/C models.

Read-only HF/Kaggle inputs; no cloud writes, labels, training or competition entry.
Not a replacement for the original exact two-T4/offline Kaggle qualification.
"""
import ast
import contextlib
import copy
import csv
import gc
import hashlib
import io
import json
import os
import resource
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

ROOT = Path('/content/rsna-final-BC-T4-20261021-v1')
COMPETITION = 'rsna-knee-abnormality-detection'
API = 'https://api.kaggle.com/v1/competitions.CompetitionApiService/'
START = time.monotonic()


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def emit(phase, **values):
    print(json.dumps({'phase': phase, 'elapsed_seconds': time.monotonic()-START,
                      **values}, allow_nan=False), flush=True)


def path_ok(name):
    p = PurePosixPath(name)
    assert name and not p.is_absolute() and '..' not in p.parts and '\\' not in name
    return p


def unpack_one(raw, name, expected):
    if raw[:4] == b'PK\x03\x04':
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            members = z.infolist()
            assert len(members) == 1
            member = members[0]
            path_ok(member.filename)
            assert member.filename in (name, PurePosixPath(name).name)
            assert member.file_size == expected
            raw = z.read(member)
    assert len(raw) == expected
    return raw


def main(config, hf_token, kaggle_token):
    import requests
    import numpy as np
    import torch
    import timm
    from huggingface_hub import HfApi, hf_hub_download
    os.umask(0o077)
    assert not ROOT.exists()
    assert torch.cuda.device_count() == 1 and 'T4' in torch.cuda.get_device_name(0)
    assert timm.__version__ == '1.0.29'
    ROOT.mkdir(mode=0o700)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_DISABLE_TELEMETRY='1')
    emit('actual_single_T4_started', torch=torch.__version__, timm=timm.__version__,
         GPU=torch.cuda.get_device_name(0), GPU_bytes=torch.cuda.get_device_properties(0).total_memory)

    def rpc(method, body):
        assert method in ('ListDataTreeFiles', 'DownloadDataFile')
        r = requests.post(API+method, json={'competitionName':COMPETITION, **body},
            headers={'Authorization':'Bearer '+kaggle_token}, timeout=(10,60), allow_redirects=False)
        assert r.status_code in (200,302), ('Kaggle HTTP', r.status_code)
        return r

    def tree(path):
        assert path == 'test_series' or path.startswith('test_series/')
        value = rpc('ListDataTreeFiles', {'path':path,'pageSize':1000}).json()
        assert not value.get('nextPageToken'), 'Unexpected pagination; do not truncate'
        return value

    def download(name, expected):
        path_ok(name)
        assert name == 'test.csv' or name.startswith('test_series/')
        assert 0 < expected < 32*2**20 and time.monotonic()-START < 1800
        r = rpc('DownloadDataFile', {'fileName':name})
        url = r.headers['Location'] if r.status_code == 302 else r.json()['url']
        parsed = urlparse(url)
        assert parsed.scheme == 'https' and parsed.hostname == 'storage.googleapis.com'
        # The bearer token is never forwarded to object storage.
        with requests.get(url, stream=True, timeout=(10,60), allow_redirects=False) as response:
            assert response.status_code == 200
            buf = bytearray()
            for block in response.iter_content(2**20):
                buf.extend(block)
                assert len(buf) <= expected + 2**20
        raw = unpack_one(bytes(buf), name, expected)
        dest = ROOT/'public'/name
        dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with dest.open('xb') as f: f.write(raw)
        return {'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}

    download('test.csv',212)
    with (ROOT/'public/test.csv').open() as f:
        uids = [r['StudyInstanceUID'] for r in csv.DictReader(f)]
    assert hashlib.sha256(json.dumps(uids,sort_keys=True,separators=(',',':')).encode()).hexdigest() == 'e743f6d8dca6eabedec2f05a98cf86ed820ef1fe8a0eb06338af5b377dc8a59b'
    top = tree('test_series')
    assert {d['name'] for d in top['directories']} == set(uids) and len(uids) == 3
    members=[]
    for study in top['directories']:
        middle = tree(study['relativeUrl'])
        assert not middle.get('files')
        for series in middle['directories']:
            leaf = tree(series['relativeUrl'])
            assert not leaf.get('directories')
            for f in leaf['files']:
                assert f['relativeUrl'].startswith('test_series/'+study['name']+'/')
                assert f['relativeUrl'].endswith('.dcm')
                members.append((f['relativeUrl'], int(f['totalBytes'])))
    assert 0 < len(members) < 10000 and len(set(n for n,_ in members)) == len(members)
    assert sum(n for _,n in members) < 2*2**30
    with ThreadPoolExecutor(max_workers=8) as pool:
        proofs=[]
        for proof in pool.map(lambda row: download(*row), members):
            proofs.append(proof)
            if len(proofs) % 100 == 0:
                emit('public_download_progress',files=len(proofs),total_files=len(members))
    (ROOT/'public-inputs.private.json').write_text(json.dumps(proofs,sort_keys=True))
    emit('complete_public_MRI_downloaded', studies=3, DICOM_files=len(proofs),
         bytes=sum(p['bytes'] for p in proofs))

    api=HfApi(token=hf_token)
    assert api.repo_info('GodlyDonuts/RSNA-Colab-Deadline-20261021',repo_type='dataset').private
    asset=hf_hub_download('GodlyDonuts/RSNA-Colab-Deadline-20261021','inputs/assets.tar',
        revision='42cb16bf2f7f8ea2559efea24af9f8af88272ff4',repo_type='dataset',token=hf_token)
    # Only the parent/source/license are extracted; no training labels or cases are read.
    needed={'parent.pt':config['parent_sha256'],'parent_source.private.py':config['parent_source_sha256'],
            'source-permission.txt':config['license_sha256']}
    with tarfile.open(asset,'r:') as archive:
        for name, expected in needed.items():
            item=archive.getmember(name)
            assert item.isfile() and not item.issym() and not item.islnk()
            assert item.size < 300*2**20
            raw=archive.extractfile(item).read()
            assert hashlib.sha256(raw).hexdigest()==expected
            with (ROOT/name).open('xb') as f:f.write(raw)
    exports=[]
    for publication in config['publications']:
        assert api.repo_info('GodlyDonuts/RSNA-Data',repo_type='dataset').private
        files={}
        for name, proof in publication['files'].items():
            path=Path(hf_hub_download('GodlyDonuts/RSNA-Data',publication['prefix']+'/'+name,
                revision=publication['revision'],repo_type='dataset',token=hf_token))
            assert path.stat().st_size==proof['bytes'] and sha(path)==proof['sha256']
            files[name]=path
        exports.append((publication,files))
    hf_token=kaggle_token=None
    for name, source in config['helpers'].items():
        assert name in ('contract.py','descriptor_cache.py')
        if name=='contract.py':
            contract={};exec(compile(source,name,'exec'),contract)
    chosen=[];sources=None
    for publication,files in exports:
        candidate=contract['candidate'](publication,files['provenance.json'].read_text(),
            files['own-source.json'].read_text(),files['source-permission.txt'].read_bytes())
        own=json.loads(files['own-source.json'].read_text())
        current={n:own[n] for n in contract['MODULES']}
        assert sources is None or current==sources
        sources=current;chosen.append(candidate)
    for name, source in sources.items():
        path_ok(name);p=ROOT/'code'/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(source)
    sys.path.insert(0,str(ROOT/'code'))
    from rsna.competitive.predictive.cases import assemble
    from rsna.competitive.predictive.dicom import load_study
    from rsna.competitive.predictive.model import Predictive, ReconstructionControl
    from rsna.competitive.predictive.infer import predict
    helper={};exec(compile(config['helpers']['descriptor_cache.py'],'descriptor_cache.py','exec'),helper)
    prepared=[]
    for i,uid in enumerate(uids):
        begin=time.monotonic();groups,stats=load_study(ROOT/'public/test_series'/uid,uid)
        decoded=time.monotonic()-begin
        groups=helper['cache_descriptors_once'](groups)
        contexts=assemble(groups,side=320,halo=8,context_cap=24,classes=12)
        record={**stats,'decode_seconds':decoded,'render_seconds':time.monotonic()-begin-decoded,
                'windows':sum(len(c['images']) for c in contexts),'regions':len(contexts[0]['targets'])}
        prepared.append((contexts,record));del groups
        emit('actual_public_inputs_prepared',study_index=i,**record)
    parent_text=(ROOT/'parent_source.private.py').read_text()
    nodes=[n for n in ast.parse(parent_text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
           and n.name in ('build_backbone','RaptorClassifier','load_model')]
    assert len(nodes)==3
    ns=dict(torch=torch,nn=torch.nn,timm=timm,gc=gc,hashlib=hashlib,INPUT_RES=320)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_parent','exec'),ns)
    with contextlib.redirect_stdout(io.StringIO()):
        parent,side=ns['load_model'](str(ROOT/'parent.pt'),'coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k',384,torch.device('cpu'))
    assert side==320
    class Encoder(torch.nn.Module):
        def __init__(self,backbone):
            super().__init__();self.backbone=backbone
            self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
            self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])
        def forward(self,x):return self.backbone((x-self.mean)/self.std)
    outcomes=[]
    for candidate,(_,files) in zip(chosen,exports):
        cls=ReconstructionControl if candidate['arm']=='B' else Predictive
        base=cls(Encoder(copy.deepcopy(parent.backbone)),parent.backbone.num_features,hidden=128)
        model=contract['validate_export'](torch.load(files['model.pt'],map_location='cpu',weights_only=True),candidate,base)
        model.cuda().eval();torch.cuda.reset_peak_memory_stats()
        measures=[]
        for i,(contexts,stats) in enumerate(prepared):
            begin=time.monotonic();cold=predict(model,contexts,precision='fp16');torch.cuda.synchronize()
            cold_s=time.monotonic()-begin
            begin=time.monotonic();low=predict(model,contexts,precision='fp16');torch.cuda.synchronize()
            warm_s=time.monotonic()-begin
            begin=time.monotonic();reference=predict(model,contexts,precision='fp32');torch.cuda.synchronize()
            rec={'study_index':i,'cold_FP16_seconds':cold_s,'warm_FP16_seconds':warm_s,
                 'FP32_seconds':time.monotonic()-begin,'probability_gap':float((low-reference).abs().max()),
                 'repeat_FP16_gap':float((cold-low).abs().max()),**stats}
            contract['check_predictions']([uids[i]],[low.cpu().tolist()])
            if candidate['arm']=='C':
                changed=[{**c,'targets':c['targets'].mean(dim=0,keepdim=True).expand_as(c['targets']).clone()} for c in contexts]
                perturbed=predict(model,changed,precision='fp32')
                rec['regional_target_mean_replacement_max_probability_change']=float((reference-perturbed).abs().max())
            measures.append(rec);emit('trained_final_measured',arm=candidate['arm'],**rec)
        outcomes.append({'arm':candidate['arm'],'model_sha256':candidate['model']['sha256'],
            'measurements':measures,'peak_GPU_bytes':torch.cuda.max_memory_allocated(),
            'host_peak_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
        del model,base;gc.collect();torch.cuda.empty_cache()
    result={'status':'actual_single_T4_final_B_C_public_MRI_diagnostic_complete','source_git':config['source_git'],
        'models':outcomes,'GPU':'single Tesla T4','torch':torch.__version__,'timm':timm.__version__,
        'studies':3,'elapsed_seconds':time.monotonic()-START,'public_input_manifest_sha256':sha(ROOT/'public-inputs.private.json'),
        'two_T4_offline_Kaggle_acceptance':False,'clinical_accuracy_or_score':False,
        'cloud_uploads_or_Kaggle_mutations':0,'training_updates':0}
    (ROOT/'result.json').write_text(json.dumps(result,sort_keys=True,allow_nan=False))
    emit('FINAL_RESULT',result=result)
