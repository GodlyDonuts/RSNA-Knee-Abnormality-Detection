"""New compiler boundaries; no original MRI/model or GPU trial is replayed."""

import base64
import hashlib

import pytest
from test_licensed_importance_recovery import inputs

from scripts.kaggle import paired_logit_recovery as m


def payload(monkeypatch):
    parent, codec, files, targets = inputs(monkeypatch)
    files.update({name: "# New synthetic module closure\n" for name in m.EXTRA})
    return parent, codec, files, targets


def test_new_worker_uses_corrected_engine_FP16_and_unchanged_full_inference(monkeypatch):
    args = payload(monkeypatch)
    source, binding = m.build(*args, permission="synthetic permission")
    values = m.literals(source)
    worker = base64.b64decode(values["WORKER"]).decode()
    assert binding["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert binding["worker_sha256"] == hashlib.sha256(worker.encode()).hexdigest()
    assert values["FILES"] == args[2]
    assert "from rsna.competitive.paired_logit_train import fit" in worker
    assert (
        "WholeStudyRaptor(PairedLogitReadout,ImportanceReadout,CheckpointedEncode,RaptorClassifier)"
        in worker
    )
    assert "Recipe(role=a.role)" in worker
    assert "torch.autocast('cuda',dtype=torch.float16)" in worker
    assert "engineering.case(i,list(range(94)),mirror)" in worker
    assert "same(full,resumed)" in source and "for epoch in (1,2,3):" in source
    assert "full['completed']==6" in source and "[1,2,3,4,5,6]" in source
    assert "identities[1]" not in source and "roles['uniform']" not in source
    assert "importance-recovery" not in worker and "paired-logit-recovery" in worker
    assert "submission.csv" in source and "not Path" in source
    assert binding["competition_entries"] == 0
    assert args[3]["uids"][0] not in source


@pytest.mark.parametrize("missing", m.EXTRA)
def test_new_source_closure_must_be_complete(monkeypatch, missing):
    parent, codec, files, targets = payload(monkeypatch)
    files.pop(missing)
    with pytest.raises(ValueError):
        m.build(parent, codec, files, targets, permission="synthetic permission")
