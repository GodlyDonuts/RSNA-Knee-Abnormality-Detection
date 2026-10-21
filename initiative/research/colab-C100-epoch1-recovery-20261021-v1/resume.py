"""Same numerical fit, exact epoch001 recovery; read-only API retries only."""
import ast
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

CONTROL = Path('/content/rsna-colab-private-control-20261021-v2')
ORIGINAL = Path('/content/rsna-independent-C100-20261021-v1/run-C100-seed20261021')
RESUMED = ORIGINAL.with_name('run-C100-seed20261021-recovery-v1')
ORIGINAL_SCRIPT_SHA = '15e17223c6eda06bf417df0fe8b64e2fd9729e8a5571ae012274974eb29267a3'
CHECKPOINT_SHA = '43c1c75920bd31356f5fad96441e43bd1317eac504a3083613a33f89b6e49fdc'
READ_ONLY = {
    ('security.OAuthService', 'IntrospectToken'),
    ('datasets.DatasetApiService', 'GetDataset'),
    ('datasets.DatasetApiService', 'GetDatasetStatus'),
    ('datasets.DatasetApiService', 'ListDatasetFiles'),
    ('datasets.DatasetApiService', 'ListDatasets'),
}


def read_retry(call, service, method, body, *, clock=time.monotonic, sleep=time.sleep):
    """Never retries writes or changes an authorization response into success."""
    end = clock() + 600
    while True:
        try:
            return call(service, method, body)
        except Exception as exc:
            message = str(exc)
            allowed = (service, method) in READ_ONLY and type(exc).__name__ == 'RemoteError'
            temporary = bool(re.fullmatch(r'Kaggle \w+ HTTP (403|408|429|500|502|503|504)', message))
            temporary |= message == f'Kaggle {method} transport/result uncertain'
            if not allowed or not temporary or clock() >= end:
                raise
            print(json.dumps({'phase': 'readonly_API_observation_retry', 'method': method,
                              'new_uploads': 0}), flush=True)
            sleep(15)


def replace_once(text, old, new):
    assert text.count(old) == 1, 'Original control source changed'
    return text.replace(old, new)


def build(text):
    assert hashlib.sha256(text.encode()).hexdigest() == ORIGINAL_SCRIPT_SHA
    text = replace_once(text, "RUN = ROOT / 'run-C100-seed20261021'", f"RUN = Path({str(RESUMED)!r})")
    text = replace_once(text, '    from colab_storage import Store, CONTROL',
                        '    from colab_storage import CONTROL\n    Store = RecoveryStore')
    text = replace_once(text, "    RUN.mkdir(mode=0o700); once(RUN / 'run-start.json', binding)",
        "    prior_binding = json.loads((ORIGINAL / 'run-start.json').read_text())\n"
        "    assert {k:v for k,v in binding.items() if k != 'started_UTC'} == {k:v for k,v in prior_binding.items() if k != 'started_UTC'}\n"
        "    binding = prior_binding\n"
        "    RUN.mkdir(mode=0o700); once(RUN / 'run-start.json', binding)\n"
        "    once(RUN / 'continuation.json', RECOVERY)\n"
        "    (RUN / 'resume-control.py').write_bytes(Path(RECOVERY['control_path']).read_bytes())")
    text = replace_once(text, '    publications = {}',
        "    from concurrent.futures import Future\n"
        "    done = Future(); done.set_result(FIRST_PUBLICATION)\n"
        "    publications = {1: done}")
    text = replace_once(text, "                 'kaggle_io.py': CONTROL / 'kaggle_io.py'}",
        "                 'kaggle_io.py': CONTROL / 'kaggle_io.py',\n"
        "                 'resume-control.py': RUN / 'resume-control.py',\n"
        "                 'continuation.json': RUN / 'continuation.json'}")
    text = replace_once(text, '        assert time.monotonic() - started < 10 * 3600 and time.time() < absolute_stop',
        "        assert time.time() < min(RECOVERY['original_ten_hour_deadline'], absolute_stop)")
    text = replace_once(text, "    state = fit(model, Cases(), recipe, binding=binding, output=RUN/'checkpoints',",
        "    state = fit(model, Cases(), recipe, binding=binding, output=RUN/'checkpoints',\n"
        "                resume=ORIGINAL/'checkpoints/epoch-001.pt',")
    text = replace_once(text, "                'all_epochs_private_verified':True, 'clinical_or_Kaggle_score':None}",
        "                'all_epochs_private_verified':True, 'clinical_or_Kaggle_score':None,\n"
        "                'execution_continuation':RECOVERY}")
    ast.parse(text)
    return text


def main(token):
    from datetime import datetime
    os.umask(0o077)
    assert not RESUMED.exists()
    sys.path.insert(0, str(CONTROL))
    import colab_storage as storage
    prior_account = storage.account

    def account(token):
        # Identity read is the only operation inside account().
        end = time.monotonic() + 600
        while True:
            try:
                remote, pub = prior_account(token)
                break
            except Exception as exc:
                if type(exc).__name__ != 'RemoteError' or time.monotonic() >= end:
                    raise
                time.sleep(15)
        original_call = remote.call
        remote.call = lambda service, method, body: read_retry(original_call, service, method, body)
        return remote, pub

    class RecoveryStore(storage.Store):
        def __init__(self, token, lease):
            # Original file bytes and exact isolated namespace lease stay unchanged.
            old = storage.account
            storage.account = account
            try:
                super().__init__(token, lease)
            finally:
                storage.account = old

    proof = json.loads((ORIGINAL / 'publication-001/reconciled-original.json').read_text())
    assert proof['status'] == 'existing_private_Colab_publication_reconciled'
    first = proof['publication']
    assert first['id'] == 12756621 and first['version'] == 1 and first['private']
    assert first['files']['full-state.private.pt.bin']['sha256'] == CHECKPOINT_SHA
    assert storage.sha(ORIGINAL / 'checkpoints/epoch-001.pt') == CHECKPOINT_SHA
    updates = [json.loads(x) for x in (ORIGINAL / 'updates.jsonl').read_text().splitlines()]
    assert [x['step'] for x in updates] == list(range(1, 490))
    binding = json.loads((ORIGINAL / 'run-start.json').read_text())
    assert storage.sha(ORIGINAL / 'run-start.json') == first['files']['run-start.json.bin']['sha256']
    control_path = str(Path(__file__).resolve())
    recovery = dict(status='exact_epoch001_resume_after_publication_observation_failure',
                    source_checkpoint_sha256=CHECKPOINT_SHA, resume_update=486,
                    discarded_unsaved_updates=[487, 488, 489], original_updates_preserved=True,
                    original_run=str(ORIGINAL), resumed_run=str(RESUMED),
                    control_path=control_path, control_sha256=storage.sha(control_path),
                    original_control_sha256=ORIGINAL_SCRIPT_SHA,
                    original_ten_hour_deadline=datetime.fromisoformat(binding['started_UTC']).timestamp()+36000,
                    initial_epoch_Dataset_id=12756621, first_epoch_reuploaded=False,
                    numerical_functions_or_recipe_changed=False)
    assert time.time() < recovery['original_ten_hour_deadline']
    ns = dict(__name__='colab_exact_recovery', __file__=str(CONTROL/'actual_colab_train.py'),
              RecoveryStore=RecoveryStore, ORIGINAL=ORIGINAL, RECOVERY=recovery, FIRST_PUBLICATION=first)
    body = build((CONTROL / 'actual_colab_train.py').read_text())
    exec(compile(body, 'exact_original_trainer_with_explicit_epoch_recovery', 'exec'), ns)
    print(json.dumps({'phase': 'exact_original_epoch_resume_admitted', **recovery}), flush=True)
    ns['main'](token)


if __name__ == '__main__':
    main(json.load(sys.stdin)['token'])
