import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('return_sender_recovery', HERE / 'run_once.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize('fault', [None, 'ambiguous', 'different_receiver', 'changed_transport', 'running_sender'])
def test_immediate_allocation_uses_exact_original_failed_jobs(tmp_path, monkeypatch, fault):
    transport = {'returncode': 1, 'stderr': 'srun: error: evc27: task 0: Exited with exit code 1\n', 'stdout': ''}
    original = tmp_path / 'return-C0-sender-transport.json'
    original.write_text(json.dumps(transport))
    rows = [
        '997029|rsna-C0-placement-return_receive|FAILED|1:0|evc33|2026-10-21T12:23:20|2026-10-21T12:23:21',
        '997030|rsna-C0-placement-return_send|FAILED|1:0|evc27|2026-10-21T12:23:20|2026-10-21T12:23:21',
    ]
    if fault == 'ambiguous':
        rows.append(rows[-1].replace('997030', '997031'))
    if fault == 'running_sender':
        rows[-1] = rows[-1].replace('FAILED|1:0', 'RUNNING|0:0')
    proof = {'status': 'actual_original_C0_sender_reconciled_from_scheduler',
             'sender_transport_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
             'original_sacct_rows': rows, 'sender': '997030'}
    (tmp_path / 'actual-C0-return-job-identity.json').write_text(json.dumps(proof))
    if fault == 'changed_transport':
        original.write_text('{}')
    inspected = {'files': {'return-intent.json': {'intent': {'CPU_job': '999999' if fault == 'different_receiver' else '997029'}}}}
    monkeypatch.setattr(module, 'HERE', tmp_path)
    monkeypatch.setattr(module, 'JOURNAL', tmp_path)
    if fault:
        with pytest.raises(AssertionError):
            module.original_sender('C0', transport, inspected)
    else:
        assert module.original_sender('C0', transport, inspected) == '997030'


def test_original_explicit_allocation_message_keeps_original_behavior():
    assert module.original_sender('B', {'stderr': 'srun: job 993619 has been allocated resources'}, {}) == '993619'
