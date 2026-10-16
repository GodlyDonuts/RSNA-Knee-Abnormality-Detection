"""Build the private, offline fixed-average versus E16 raw-outer evaluation."""

from __future__ import annotations

import ast
import base64
import gzip
import hashlib
import json
import subprocess
from pathlib import Path


def build(prepared, epoch, destination: Path):
    if (
        prepared["status"] != "actual_CPU_fixed_average_paired_outer_component_prepared"
        or prepared["outer_studies"] != 1156
    ):
        raise ValueError("Original complete outer population and actual CPU component required")
    if (
        epoch["epoch"] != 16
        or epoch["checkpoint_sha256"] != prepared["source_checkpoint_sha256"][-1]
    ):
        raise ValueError("Pair must compare the fixed average with its unchanged original E16")
    ref = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", ref, "rsna"], text=True
    ).splitlines()
    files = {
        n: subprocess.check_output(["git", "show", ref + ":" + n]).decode()
        for n in names
        if n.endswith(".py")
    }
    required = {"compact_pair_runtime", "compact_runtime", "compact_t4_probe"}
    pending = list(required)
    while pending:
        name = pending.pop()
        path = Path("scripts/kaggle") / (name + ".py")
        text = subprocess.check_output(["git", "show", ref + ":" + str(path)], text=True)
        files[str(path)] = text
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.ImportFrom) and node.module:
                dependency = node.module.split(".")[0]
                if (path.parent / (dependency + ".py")).exists() and dependency not in required:
                    required.add(dependency)
                    pending.append(dependency)
    raw = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    assert len(raw) < 8 * 2**20
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    entry = Path("scripts/kaggle/compact_t4_entry.py").read_text()
    entry = entry[: entry.rfind("\nif __name__")]
    body = r"""
import base64,gzip
bundle,python,env=prepare_runtime('1e6345bd3d2906271ecfbe6ab0fce146f7f8f8e0e5f2ca0ba7249a559c011075')
raw=gzip.decompress(base64.b64decode(SOURCE_B64));assert len(raw)<8*2**20 and hashlib.sha256(raw).hexdigest()==PROJECTION_SHA
source=Path('/kaggle/working/paired-research-source');source.mkdir(mode=0o700)
for name,text in json.loads(raw).items():
 p=source/name;assert p.resolve().is_relative_to(source.resolve()) and p.suffix=='.py';p.parent.mkdir(mode=0o700,parents=True,exist_ok=True);p.write_text(text)
env['PYTHONPATH']=str(source)
components={}
for directory,children,names in os.walk('/kaggle/input',followlinks=False):
 p=Path(directory)
 if p.name=='rsna-knee-abnormality-detection' or {'train_series','test_series'}&set(children):children[:]=[];continue
 if len(p.relative_to('/kaggle/input').parts)>=8:children[:]=[]
 if 'component-manifest.json' in names:
  h=sha(p/'component-manifest.json')
  if h in (AVERAGE_MANIFEST,E16_MANIFEST):
   assert h not in components;components[h]=p
assert set(components)=={AVERAGE_MANIFEST,E16_MANIFEST}
for h,p in components.items():
 manifest=json.loads((p/'component-manifest.json').read_text())
 for n,row in manifest['files'].items():
  f=p/n;assert not f.is_symlink() and f.resolve().is_relative_to(p.resolve()) and f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
average=components[AVERAGE_MANIFEST];epoch=components[E16_MANIFEST]
provenance=json.loads((average/'provenance.json').read_text());assert provenance['outer_count']==1156 and provenance['checkpoint_sha256']==AVERAGE_SHA
reference=json.loads((epoch/'reference.json').read_text());assert reference['checkpoint_sha256']==E16_SHA
alias=Path('/tmp/rsna-paired-outer-images');alias.mkdir(mode=0o700)
competition=Path('/kaggle/input/competitions/rsna-knee-abnormality-detection')
if not competition.is_dir():competition=Path('/kaggle/input/rsna-knee-abnormality-detection')
(alias/'test_series').symlink_to(competition/'train_series',target_is_directory=True)
helper=alias/'select.py';helper.write_text("import hashlib,json,sys,pandas as pd\nfrom pathlib import Path\np=Path(sys.argv[1]);r=Path(sys.argv[2]);o=Path(sys.argv[3]);expected=sys.argv[4];ids=pd.read_csv(r,dtype=str).StudyInstanceUID.tolist();assert len(ids)==len(set(ids))==1156 and hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==expected;f=pd.read_csv(p/'train_series.csv');f=f[f.StudyInstanceUID.isin(ids)];assert set(f.StudyInstanceUID)==set(ids);pd.DataFrame({'StudyInstanceUID':ids}).to_csv(o/'test.csv',index=False);f.to_csv(o/'test_series.csv',index=False)\n")
subprocess.run([str(python),'-B',str(helper),str(competition),str(average/'outer-images.csv'),str(alias),provenance['roster_order_sha256']],env=env,check=True,timeout=90)
command=[str(python),'-B',str(source/'scripts/kaggle/compact_pair_runtime.py'),'--checkpoint',str(average/'average.pt'),'--checkpoint-sha256',AVERAGE_SHA,'--source-shas',*SOURCE_SHAS,'--epoch-checkpoint',str(epoch/'checkpoint.pt'),'--epoch-sha256',E16_SHA,'--fixture',str(bundle/'fixture.pt'),'--fixture-sha256',FIXTURE_SHA,'--average-reference',str(average/'average-reference.json'),'--average-reference-sha256',AVERAGE_REFERENCE_SHA,'--epoch-reference',str(epoch/'reference.json'),'--epoch-reference-sha256',sha(epoch/'reference.json'),'--series',str(alias/'test_series.csv'),'--test-studies',str(alias/'test.csv'),'--dicom-root',str(alias),'--scratch','/kaggle/working/paired-bounded-scratch','--output','/kaggle/working/private-average-outer.csv','--epoch-output','/kaggle/working/private-E16-outer.csv','--max-seconds','20700']
run_bounded_native(command,bundle,env,21000)
p=Path('/kaggle/working/private-average-outer.paired.json');result=json.loads(p.read_text())
assert result['status']=='paired_late_average_E16_raw_MRI_complete' and result['studies']==1156 and result['test_roster_sha256']==provenance['roster_order_sha256']
assert result['checkpoint_sha256']==AVERAGE_SHA and result['epoch_sha256']==E16_SHA and result['competition_submission'] is False and not Path('/kaggle/working/submission.csv').exists()
result.update(source_projection_sha256=PROJECTION_SHA,average_component_manifest_sha256=AVERAGE_MANIFEST,epoch_component_manifest_sha256=E16_MANIFEST,clinical_accuracy_claim=False,private_predictions_retained=True)
Path('/kaggle/working/paired-outer-accepted.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
"""
    substitutions = {
        "SOURCE_B64": encoded,
        "PROJECTION_SHA": digest,
        "AVERAGE_MANIFEST": prepared["component_manifest_sha256"],
        "E16_MANIFEST": epoch["component_manifest_sha256"],
        "AVERAGE_SHA": prepared["checkpoint_sha256"],
        "E16_SHA": epoch["checkpoint_sha256"],
        "SOURCE_SHAS": prepared["source_checkpoint_sha256"],
        "FIXTURE_SHA": prepared["fixture_sha256"],
        "AVERAGE_REFERENCE_SHA": prepared["average_reference_sha256"],
    }
    for token, value in substitutions.items():
        body = body.replace(token, repr(value))
    text = entry + "\n" + body
    compile(text, "paired_outer_notebook", "exec")
    if destination.exists() or len(text.encode()) >= 1024**2:
        raise ValueError("Distinct bounded notebook source required")
    destination.write_text(text)
    return {
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_projection_sha256": digest,
        "source_git": ref,
        "source_bytes": len(text.encode()),
        "outer_studies": 1156,
        "kernel_slug": "godlydonuts/rsna-fixedavg-e16-outer-" + prepared["checkpoint_sha256"][:12],
        "competition_submission": False,
    }
