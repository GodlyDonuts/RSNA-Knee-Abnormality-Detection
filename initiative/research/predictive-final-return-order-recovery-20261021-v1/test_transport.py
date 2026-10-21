import importlib.util
import io
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('return_order_recovery', Path(__file__).with_name('transport.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def objects(tmp_path):
    paths, files = {}, {}
    # The original sender put terminal first; canonical JSON makes it last.
    for index, name in enumerate(['terminal.json'] + sorted(m.NAMES - {'terminal.json'})):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (str(index) + name).encode()
        path.write_bytes(raw)
        path.chmod(0o600)
        paths[name] = path
        files[name] = {'bytes': len(raw), 'sha256': m.sha(raw), 'stat': m.signature(path)}
    return paths, files


def test_original_metadata_body_order_mismatch_is_reproduced(tmp_path):
    paths, files = objects(tmp_path)
    decoded = json.loads(m.canonical({'files': files}))['files']
    assert list(files)[0] == 'terminal.json'
    assert list(decoded)[0] == 'publication/epoch-006/manifest.json'
    original = b''.join(paths[name].read_bytes() for name in files)
    expected = decoded[next(iter(decoded))]
    assert m.sha(original[:expected['bytes']]) != expected['sha256']


def test_sorted_seven_object_transfer_roundtrip(tmp_path):
    paths, files = objects(tmp_path / 'source')
    stream = io.BytesIO()
    m.write_frame(stream, {'files': files})
    m.send_objects(stream, paths, files)
    stream.seek(0)
    header = m.read_frame(stream)
    m.receive_objects(stream, tmp_path / 'destination', header['files'])
    for name in m.NAMES:
        assert (tmp_path / 'destination' / name).read_bytes() == paths[name].read_bytes()


def test_changed_source_refused(tmp_path):
    paths, files = objects(tmp_path)
    paths['terminal.json'].write_bytes(b'changed')
    with pytest.raises(ValueError, match='stat changed'):
        m.send_objects(io.BytesIO(), paths, files)


def test_corrupt_bytes_refused_without_acceptance(tmp_path):
    paths, files = objects(tmp_path / 'source')
    stream = io.BytesIO()
    m.send_objects(stream, paths, files)
    raw = bytearray(stream.getvalue())
    raw[0] ^= 1
    with pytest.raises(ValueError, match='SHA mismatch'):
        m.receive_objects(io.BytesIO(raw), tmp_path / 'destination', files)
    assert not (tmp_path / 'destination/return-verified.json').exists()


def test_duplicate_destination_refused(tmp_path):
    paths, files = objects(tmp_path / 'source')
    stream = io.BytesIO()
    m.send_objects(stream, paths, files)
    raw = stream.getvalue()
    m.receive_objects(io.BytesIO(raw), tmp_path / 'destination', files)
    with pytest.raises(FileExistsError):
        m.receive_objects(io.BytesIO(raw), tmp_path / 'destination', files)


def test_truncated_stream_and_trailing_bytes_refused(tmp_path):
    paths, files = objects(tmp_path / 'source')
    stream = io.BytesIO()
    m.send_objects(stream, paths, files)
    raw = stream.getvalue()
    for suffix, content in [('truncated', raw[:-1]), ('extra', raw + b'x')]:
        with pytest.raises(ValueError):
            m.receive_objects(io.BytesIO(content), tmp_path / suffix, files)


def test_symlink_source_refused(tmp_path):
    original = tmp_path / 'original'
    original.write_bytes(b'x')
    original.chmod(0o600)
    link = tmp_path / 'link'
    link.symlink_to(original)
    with pytest.raises(ValueError, match='regular object'):
        m.signature(link)


def test_mismatched_or_incomplete_header_refused():
    v = {'terminal': {'arm': 'C'}, 'publications': [{'files': {}}], 'job': {'binding_sha256': 'a' * 64}}
    header = {'terminal': {'arm': 'B'}, 'publications': v['publications'], 'binding_sha256': 'a' * 64}
    with pytest.raises(ValueError, match='terminal changed'):
        m.check_header(header, v)
    header['terminal'] = v['terminal']
    header.update(files={}, ordered_names=[])
    with pytest.raises(ValueError, match='seven original'):
        m.check_header(header, v)
