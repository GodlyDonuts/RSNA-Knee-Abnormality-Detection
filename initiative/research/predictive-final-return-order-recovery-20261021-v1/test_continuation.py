"""The deadline helper must not mutate or interrupt healthy/completed originals."""
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent


def module(tmp_path):
    spec=importlib.util.spec_from_file_location('C0_deadline_test',HERE/'continue_C0.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OLD=tmp_path/'old';m.OLD.mkdir()
    m.ORIGINAL=tmp_path/'original';m.ORIGINAL.mkdir()
    m.JOURNAL=tmp_path/'journal'
    return m


def original_git(command, **kwargs):
    assert command[:2]==['git','show']
    name=command[2].split(':',1)[1].rsplit('/',1)[1]
    return (HERE/name).read_bytes()


def test_existing_return_never_calls_remote(tmp_path):
    m=module(tmp_path);(m.OLD/'actual-owner-return-C0.json').write_text('{}')
    with patch.object(m.subprocess,'check_output',side_effect=original_git), \
         patch.object(m.subprocess,'run',side_effect=AssertionError('remote mutation')), \
         patch.object(m.time,'time',return_value=0):
        m.main('f'*40,'unused')
    assert (m.JOURNAL/'already-returned.json').exists()


def test_healthy_owner_is_only_waited_on(tmp_path):
    m=module(tmp_path)
    def healthy_owner_finishes(_):
        (m.OLD/'actual-owner-return-C0.json').write_text('{}')
    with patch.object(m.subprocess,'check_output',side_effect=original_git), \
         patch.object(m.subprocess,'run',side_effect=AssertionError('remote mutation')), \
         patch.object(m.time,'time',return_value=0), \
         patch.object(m.time,'sleep',side_effect=healthy_owner_finishes):
        m.main('f'*40,'unused')
    assert not (m.JOURNAL/'destination-observation-intent.json').exists()


def test_expired_window_does_not_launch_any_phase(tmp_path):
    m=module(tmp_path)
    with patch.object(m.subprocess,'check_output',side_effect=original_git), \
         patch.object(m.subprocess,'run',side_effect=AssertionError('remote mutation')), \
         patch.object(m.time,'time',return_value=10**12):
        with pytest.raises(TimeoutError):m.main('f'*40,'unused')
    assert not (m.JOURNAL/'destination-observation-intent.json').exists()
