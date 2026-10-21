"""Derive the original Colab numerical loop with private Kaggle preservation."""

import ast
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent


def build():
    spec = importlib.util.spec_from_file_location(
        "Colab_training_transport_derivation", HERE / "derive.py"
    )
    d = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(d)
    text = d.original_files()["colab_train.py"]
    text = d.once(
        text,
        "    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download",
        "    from colab_storage import Store, CONTROL",
    )
    text = d.once(
        text,
        "    api = HfApi(token=token)\n    assert api.repo_info(REPO, repo_type='dataset').private\n    assert not list(api.get_paths_info(REPO, paths=[PREFIX], repo_type='dataset'))",
        "    lease = json.loads((CONTROL / 'checkpoint-lease.private.json').read_text())\n"
        "    store = Store(token, lease)\n"
        "    assert lease['input_manifest_sha256'] == inputs['downloaded_manifest_sha256']",
    )
    text = d.once(
        text,
        "'started_UTC': datetime.now(timezone.utc).isoformat()}",
        "'storage_control_git': lease['storage_control_git'],\n"
        "        'storage_control_sha256': lease['code_sha256'],\n"
        "        'started_UTC': datetime.now(timezone.utc).isoformat()}",
    )
    text = d.once(
        text,
        "        assert api.repo_info(REPO, repo_type='dataset').private\n        assert not list(api.get_paths_info(REPO, paths=[prefix], repo_type='dataset'))\n",
        "",
    )
    text = d.once(
        text,
        "'colab_train.py': ROOT / 'colab_train.py'}",
        "'colab_train.py': CONTROL / 'actual_colab_train.py',\n"
        "                 'colab_storage.py': CONTROL / 'colab_storage.py',\n"
        "                 'kaggle_io.py': CONTROL / 'kaggle_io.py'}",
    )
    start = text.index("        proofs = {n: ")
    end = text.index("\n    def seal():", start)
    text = (
        text[:start]
        + "        return store.publish(f'epoch-{epoch:03d}', files, destination)\n"
        + text[end:]
    )
    start = text.index("    api.create_commit(REPO, ")
    end = text.index("    print(json.dumps(terminal), flush=True)", start)
    text = (
        text[:start]
        + "    destination=RUN/'terminal-publication';destination.mkdir(mode=0o700)\n"
        + "    store.publish('terminal', {'terminal.json':RUN/'terminal.json'}, destination)\n"
        + text[end:]
    )
    ast.parse(text)
    return text
