import importlib.util
from pathlib import Path
import numpy as np
import pytest

s = importlib.util.spec_from_file_location('readout_diag', Path(__file__).with_name('diagnostic.py'))
d = importlib.util.module_from_spec(s); s.loader.exec_module(d)


def test_selection_is_order_independent_and_label_blind():
    ids = [str(i) for i in range(150)]
    reverse = ids[::-1]
    assert [ids[i] for i in d.select(ids)] == [reverse[i] for i in d.select(reverse)]
    assert len(set(d.select(ids))) == 128
    with pytest.raises(AssertionError): d.select(['x']*150)


def test_masked_soft_BCE_and_direction():
    q = np.ones((4,12)); mask = np.ones((4,12), dtype=bool)
    q[0,0] = np.nan; mask[0,0] = False
    r = d.summarize(q, mask, np.full(q.shape,.5), np.full(q.shape,.75), np.zeros(q.shape))
    assert r['observed_labels'] == 47
    assert r['context_BCE'] == pytest.approx(np.log(2))
    assert r['full_BCE'] == pytest.approx(-np.log(.75))
    assert r['full_minus_context_BCE'] < 0
    assert r['probability_shift_mean_abs'] == pytest.approx(.25)
    assert not r['independent_accuracy_claim']


def test_missing_target_and_zero_one_boundaries():
    q = np.tile([0.,1.], (4,6)); mask = np.ones(q.shape,dtype=bool); mask[:,0]=False
    r=d.summarize(q,mask,q,q,np.zeros(q.shape))
    assert r['targets'][0]['context_BCE'] is None
    assert r['full_minus_context_BCE'] == 0 and np.isfinite(r['full_BCE'])


def test_rejects_nonfinite_prediction():
    q=np.zeros((4,12)); p=q.copy(); p[0,0]=np.nan
    with pytest.raises(AssertionError): d.summarize(q,np.ones(q.shape,dtype=bool),p,q,q)
