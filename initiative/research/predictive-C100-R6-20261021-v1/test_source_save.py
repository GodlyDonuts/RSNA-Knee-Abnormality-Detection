"""Exercise source-only account lifecycle, especially failures after mutation."""

import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
s = importlib.util.spec_from_file_location("ensemble_source_save_tested", HERE / "source_save.py")
M = importlib.util.module_from_spec(s)
s.loader.exec_module(M)
c = importlib.util.spec_from_file_location("ensemble_original_compiler_tested", HERE / "compile.py")
C = importlib.util.module_from_spec(c)
c.loader.exec_module(C)


class Remote:
    def __init__(self, request, score="0.950", failure=False):
        self.request, self.failure = request, failure
        self.entry = dict(ref=57491555, status="COMPLETE", publicScore=score, errorDescription=None)
        self.mutations = []

    def submissions(self):
        return [self.entry]

    def account(self):
        return dict(entered=True, account="godlydonuts", num_allowed_now=1)

    def dataset(self, slug):
        assert slug == M.MODEL_DATASET
        return dict(id=12762386, currentVersionNumber=1)

    def require_absent_kernel(self, slug):
        assert slug == M.SLUG
        return dict(method="synthetic_known_absence")

    def call(self, service, method, body):
        assert service == "kernels.KernelsApiService"
        if method == "GetAcceleratorQuotaStatistics":
            return dict(gpuQuota=dict(totalTimeAllowed="2500s", timeUsed="100s"))
        assert method == "SaveKernel"
        assert body["kernelExecutionType"] == "QUICK_SAVE"
        self.mutations.append(body)
        if self.failure:
            raise TimeoutError("simulated result uncertainty after accepting request")
        return dict(ref="/code/" + M.SLUG, versionNumber=1, kernelId=123)

    def kernel(self, method, slug, version):
        assert (method, slug, version) == ("GetKernel", M.SLUG, 1)
        return dict(metadata={**self.request, "id": 123}, blob=dict(source=self.request["text"]))


def fixture(tmp_path, monkeypatch):
    source = "synthetic child source; not executable clinical data"
    monkeypatch.setattr(M, "SOURCE_SHA", hashlib.sha256(source.encode()).hexdigest())
    metadata = dict(
        id=139271768,
        currentVersionNumber=2,
        datasetDataSources=sorted(M.DATASETS),
        kernelDataSources=[M.KERNEL],
        competitionDataSources=["rsna-knee-abnormality-detection"],
        dockerImage=M.DOCKER,
    )
    prepared = dict(
        source_sha256=M.SOURCE_SHA,
        source_git=M.PREPARATION,
        CPU_job="1002213",
        actual_combined_GPU_run=False,
    )
    request = M.make_request(source, prepared, metadata)
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY,value TEXT)")

    def new_json(path, value):
        with path.open("x") as f:
            json.dump(value, f)

    w = SimpleNamespace(
        new_json=new_json,
        object_hash=lambda v: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest(),
    )
    return source, prepared, metadata, request, db, w


def test_source_only_requires_exact_originals_and_keeps_all_inputs(tmp_path, monkeypatch):
    source, p, m, q, db, w = fixture(tmp_path, monkeypatch)
    assert q["kernelExecutionType"] == "QUICK_SAVE"
    assert set(q["datasetDataSources"]) == M.DATASETS | {M.MODEL_DATASET}
    assert q["isPrivate"] and not q["enableInternet"] and q["sessionTimeoutSeconds"] == 32400
    with pytest.raises(AssertionError):
        M.make_request(source + "changed", p, m)
    with pytest.raises(AssertionError):
        M.make_request(source, p, {**m, "currentVersionNumber": 3})
    with pytest.raises(AssertionError):
        M.make_request(source, p, {**m, "datasetDataSources": []})


@pytest.mark.parametrize(
    "entry",
    [
        dict(ref=57491555, status=None, publicScore=None),
        dict(ref=57491555, status="COMPLETE", publicScore=None, errorDescription="failed"),
        dict(ref=57491555, status="COMPLETE", publicScore="0.948"),
    ],
)
def test_no_account_mutation_or_durable_source_intent_before_score(tmp_path, monkeypatch, entry):
    *_, q, db, w = fixture(tmp_path, monkeypatch)
    r = Remote(q)
    r.entry = entry
    out = M.save_source(w, r, db, q, C, "test-pin", root=tmp_path)
    assert out["status"] == "waiting_actual_C100_qualifying_score_no_mutation"
    assert not r.mutations and not list(tmp_path.iterdir())
    assert db.execute("SELECT count(*) FROM meta").fetchone()[0] == 0


def test_one_save_then_reconciliation_never_second_mutation(tmp_path, monkeypatch):
    *_, q, db, w = fixture(tmp_path, monkeypatch)
    r = Remote(q)
    out = M.save_source(w, r, db, q, C, "test-pin", root=tmp_path)
    assert out["status"] == "C100_R6_private_source_only_saved_requires_native_input_pins"
    assert out["input_versions_pinned"] is False and out["batch_GPU_execution"] is False
    assert len(r.mutations) == 1
    again = M.save_source(w, r, db, q, C, "test-pin", root=tmp_path)
    assert again["status"] == "existing_original_source_operation_do_not_repeat"
    assert len(r.mutations) == 1 and again["actual"] == out


def test_uncertain_save_retains_durable_intent_and_cannot_repeat(tmp_path, monkeypatch):
    *_, q, db, w = fixture(tmp_path, monkeypatch)
    r = Remote(q, failure=True)
    with pytest.raises(TimeoutError):
        M.save_source(w, r, db, q, C, "test-pin", root=tmp_path)
    assert (tmp_path / "source-save-intent.json").is_file()
    assert not (tmp_path / "source-save-response.json").exists()
    row = json.loads(db.execute("SELECT value FROM meta WHERE key=?", (M.KEY,)).fetchone()[0])
    assert row["state"] == "C100_R6_source_only_intent"
    out = M.save_source(w, r, db, q, C, "test-pin", root=tmp_path)
    assert out["original_state"] == "C100_R6_source_only_intent" and len(r.mutations) == 1
