"""Build one diagnostic-only offline dense T4 entry from immutable source."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE = "cded0713"


def build(prepared: dict, destination: Path):
    if (
        prepared.get("status") != "actual_neighbor_dense_T4_diagnostic_CPU_component_prepared"
        or prepared.get("diagnostic_only") is not True
        or prepared.get("competition_eligible") is not False
    ):
        raise ValueError("Actual diagnostic-only CPU component required")
    for key in ("checkpoint_sha256", "component_manifest_sha256"):
        value = prepared.get(key)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Exact whole-byte identities required")
    ref = subprocess.check_output(["git", "rev-parse", SOURCE], text=True).strip()
    names = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", ref, "rsna"], text=True
    ).splitlines()
    names += ["scripts/kaggle/compact_dense_t4_probe.py", "scripts/kaggle/compact_t4_probe.py"]
    files = {
        n: subprocess.check_output(["git", "show", ref + ":" + n]).decode()
        for n in names
        if n.endswith(".py")
    }
    raw = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    assert len(raw) < 8 * 2**20
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    entry = Path("scripts/kaggle/compact_t4_entry.py").read_text()
    entry = entry[: entry.rfind("\nif __name__")]
    body = (
        r"""
import base64,gzip
bundle,python,env=prepare_runtime('1e6345bd3d2906271ecfbe6ab0fce146f7f8f8e0e5f2ca0ba7249a559c011075')
raw=gzip.decompress(base64.b64decode(SOURCE_B64));assert len(raw)<8*2**20 and hashlib.sha256(raw).hexdigest()==PROJECTION_SHA
source=Path('/kaggle/working/dense-diagnostic-source');source.mkdir(mode=0o700)
for name,text in json.loads(raw).items():
 p=source/name;assert p.resolve().is_relative_to(source.resolve()) and p.suffix=='.py';p.parent.mkdir(mode=0o700,parents=True,exist_ok=True);p.write_text(text)
env['PYTHONPATH']=str(source)
found=[];seen=0
for directory,children,names in os.walk('/kaggle/input',followlinks=False):
 p=Path(directory);seen+=1;assert seen<=2048
 if p.name=='rsna-knee-abnormality-detection' or {'train_series','test_series'}&set(children):children[:]=[];continue
 if len(p.relative_to('/kaggle/input').parts)>=8:children[:]=[]
 if 'component-manifest.json' in names and sha(p/'component-manifest.json')==MANIFEST_SHA:found.append(p)
assert len(found)==1
output=Path('/kaggle/working/dense-T4-accepted.json')
command=[str(python),'-B',str(source/'scripts/kaggle/compact_dense_t4_probe.py'),'--component',str(found[0]),'--manifest-sha256',MANIFEST_SHA,'--output',str(output)]
run_bounded_native(command,bundle,env,1800)
result=json.loads(output.read_text());assert result['status']=='actual_neighbor_dense30_two_T4_precision_and_timing_accepted'
assert result['checkpoint_sha256']==CHECKPOINT_SHA and result['diagnostic_only'] is True and result['competition_eligible'] is False
assert not Path('/kaggle/working/submission.csv').exists()
result.update(source_projection_sha256=PROJECTION_SHA,component_manifest_sha256=MANIFEST_SHA)
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
""".replace("SOURCE_B64", repr(encoded))
        .replace("PROJECTION_SHA", repr(digest))
        .replace("MANIFEST_SHA", repr(prepared["component_manifest_sha256"]))
        .replace("CHECKPOINT_SHA", repr(prepared["checkpoint_sha256"]))
    )
    text = entry + "\n" + body
    compile(text, "dense_diagnostic_notebook", "exec")
    if destination.exists():
        raise ValueError("Original entry exists; no overwrite")
    destination.write_text(text)
    return {
        "source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source_projection_sha256": digest,
        "source_bytes": len(text.encode()),
        "source_git": ref,
        "kernel_slug": "godlydonuts/rsna-neighbor-dense30-t4-" + prepared["checkpoint_sha256"][:12],
        "checkpoint_sha256": prepared["checkpoint_sha256"],
        "diagnostic_only": True,
        "competition_submission": False,
    }
