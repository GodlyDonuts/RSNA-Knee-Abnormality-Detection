"""Reuse original archives and unchanged numerical inputs, changing transport only."""

import ast
import hashlib
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ORIGINAL = "initiative/research/colab-independent-C100-20261021-v1/"
PIN = "109c9965c52d16684f0e167d038c3e2b30e998f2"
SCIENCE = "94a3921aefdf333624caf41aff77aa10b4a0c57e"
SLUG = "godlydonuts/rsna-colab-c100-contexts-20261021-v1"


def frozen(path, pin=PIN):
    return subprocess.check_output(["git", "show", pin + ":" + path], cwd=REPO, text=True)


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Original transport derivation point differs")
    return text.replace(old, new)


def bridge():
    text = frozen(ORIGINAL + "resume_bridge.py")
    text = once(
        text,
        "assert v.pop('storage_handoff_resolved') is True",
        "assert v.pop('use_private_Kaggle_storage') is True\n"
        "    transport = ROOT / 'private-kaggle-transport-v1'\n"
        "    assert not transport.exists(), 'Reconcile the original handoff; no duplicate'\n"
        "    transport.mkdir(mode=0o700)\n"
        "    upload_dir = transport / 'upload'\n"
        "    upload_dir.mkdir(mode=0o700)",
    )
    text = once(
        text, "'storage-continuation-intent.json'", "'kaggle-transport-reconciliation.json'"
    )
    start = text.index("    def upload(path):\n")
    end = text.index("    def add(tar, name,", start)
    replacement = """    def upload(path):
        h, size = digest(path), path.stat().st_size
        assert size < 8 * GIB
        name = 'inputs/' + path.name
        proof_path = ROOT / (path.name + '.uploaded.json')
        if proof_path.exists():
            proof = json.loads(proof_path.read_text())
            assert proof['path'] == name and proof['bytes'] == size and proof['sha256'] == h
            manifest['archives'].append({**proof, 'provider': 'HF'})
            print(json.dumps({'phase':'reuse_completed_HF_archive', 'name':path.name,
                              'bytes':size}), flush=True)
            return
        # Opaque extension prevents Dataset ingestion from expanding the TAR.
        remote_name = path.name + '.bin'
        os.link(path, upload_dir / remote_name)
        manifest['archives'].append({'provider':'Kaggle', 'path':name,
            'remote_name':remote_name, 'bytes':size, 'sha256':h, 'private':True,
            'slug':KAGGLE_SLUG, 'version':1,
            'independent_Colab_download_verification_required':True})
        print(json.dumps({'phase':'remaining_archive_prepared', 'name':remote_name,
                          'bytes':size}), flush=True)

"""
    text = text[:start] + replacement + text[end:]
    start = text.index("    save(ROOT / 'manifest.private.json', manifest)\n")
    end = text.index("\n\nif __name__ == '__main__':", start)
    tail = """    assert sum(a['provider'] == 'HF' for a in manifest['archives']) == 7
    assert sum(a['bytes'] for a in manifest['archives'] if a['provider'] == 'HF') == 26024355840
    manifest['transport'] = {'kind':'original_HF_plus_private_Kaggle',
        'control_git':v['bridge_control_git'], 'Kaggle_slug':KAGGLE_SLUG,
        'HF_completed_uploads_repeated':0, 'new_GPU_or_competition_mutations':0}
    save(transport / 'manifest.private.json', manifest)
    os.link(transport / 'manifest.private.json', upload_dir / 'input-manifest.bin')
    proofs = {p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in upload_dir.iterdir()}
    assert sum(p['bytes'] for p in proofs.values()) < 48 * GIB
    result = create_private_inputs(v.pop('Kaggle_token'), KAGGLE_SLUG, upload_dir,
                                   proofs, transport, v['bridge_control_git'])
    save(transport / 'created.json', result)
    print(json.dumps(result), flush=True)
"""
    text = text[:start] + tail + text[end:]
    text = once(text, "GIB = 2**30", "GIB = 2**30\nKAGGLE_SLUG = " + repr(SLUG))
    ast.parse(text)
    return text


def sources():
    files = {
        name: frozen("scripts/kaggle/" + name)
        for name in ("workflow.py", "components.py", "publication.py")
    }
    files["kaggle_io.py"] = (HERE / "kaggle_io.py").read_text()
    return files


def original_files():
    return {n: frozen(ORIGINAL + n, SCIENCE) for n in ("colab_download.py", "colab_train.py")}


def source_receipt():
    return {
        "original_transport_git": PIN,
        "numerical_Colab_git": SCIENCE,
        "derived_bridge_sha256": hashlib.sha256(bridge().encode()).hexdigest(),
        "complete_original_HF_archives_reused": 7,
        "private_Kaggle_slug": SLUG,
        "new_GPU_jobs": 0,
    }
