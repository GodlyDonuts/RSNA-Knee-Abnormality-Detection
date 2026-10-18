"""Only new control deltas; no old MRI/optimizer replay."""
from pathlib import Path

D = Path(__file__).resolve().parents[1] / 'encoder-lr-comparison-20261018-v1'


def test_sole_remote_delta_is_dependency_case_and_distinct_execution_receipt():
    before = (D / 'cloud-v2.py.txt').read_text()
    expected = before.replace("(N/'execution-v2.json')", "(N/'execution-v3.json')")
    old = "dependencies.append(json.loads((N/('epoch-'+str(epoch-1))/'push-intent.json').read_text())['slug'])"
    expected = expected.replace(old, old[:-1]+'.lower())')
    assert (D / 'cloud-v3.py.txt').read_text() == expected
    compile(expected, 'distinct-normalization-control', 'exec')


def test_successor_starts_at_original_E2_status_without_repush_or_E1_republication():
    source = (D / 'finish_v4_once.py.txt').read_text()
    assert 'for epoch in (2,3):' in source and 'if epoch>2:' in source
    assert "phase='status' if epoch==2 else 'push'" in source
    assert "'maximum_new_GPU_pushes':1" in source
    assert 'run_phase_v3.py.txt' in source and 'run_phase_v2.py.txt' not in source
    assert 'owner-v4-intent.json' in source
    compile(source, 'distinct-finite-original-E2-owner', 'exec')
