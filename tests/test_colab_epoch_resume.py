import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    s = importlib.util.spec_from_file_location(name, ROOT / path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


m = load('initiative/research/colab-C100-epoch1-recovery-20261021-v1/resume.py', 'colab_resume_test')


def test_only_wrapper_changes_exact_original_fit_and_state_recovery():
    original = load('initiative/research/colab-private-kaggle-handoff-20261021-v1/training.py', 'colab_original_test').build()
    assert hashlib.sha256(original.encode()).hexdigest() == m.ORIGINAL_SCRIPT_SHA
    recovered = m.build(original)
    assert "resume=ORIGINAL/'checkpoints/epoch-001.pt'" in recovered
    assert 'binding = prior_binding' in recovered
    assert 'publications = {1: done}' in recovered
    assert "'resume-control.py': RUN / 'resume-control.py'" in recovered
    assert original[original.index('    class Cases:'):original.index('    publication_pool =')] == recovered[recovered.index('    class Cases:'):recovered.index('    publication_pool =')]
    with pytest.raises(AssertionError):
        m.build(original + '\n')


class RemoteError(RuntimeError):
    pass


def test_transient_read_is_bounded_and_retried():
    calls, waits = [], []
    def call(*args):
        calls.append(args)
        if len(calls) == 1:
            raise RemoteError('Kaggle GetDatasetStatus HTTP 403')
        return {'status': 'READY'}
    assert m.read_retry(call, 'datasets.DatasetApiService', 'GetDatasetStatus', {}, clock=lambda:0, sleep=waits.append) == {'status':'READY'}
    assert len(calls) == 2 and waits == [15]


@pytest.mark.parametrize('method', ['CreateDataset', 'CreateDatasetVersion', 'SaveKernel', 'SubmitToCompetition'])
def test_writes_are_never_retried(method):
    waits = []
    def call(*args):
        raise RemoteError(f'Kaggle {method} HTTP 503')
    with pytest.raises(RemoteError):
        m.read_retry(call, 'datasets.DatasetApiService', method, {}, sleep=waits.append)
    assert not waits


def test_other_errors_and_expired_observation_stop():
    clock = iter([0, 601])
    def call(*args):
        raise RemoteError('Kaggle GetDatasetStatus HTTP 403')
    with pytest.raises(RemoteError):
        m.read_retry(call, 'datasets.DatasetApiService', 'GetDatasetStatus', {}, clock=lambda:next(clock))
    def bad(*args):
        raise ValueError('Binding failed')
    with pytest.raises(ValueError):
        m.read_retry(bad, 'datasets.DatasetApiService', 'GetDatasetStatus', {})
