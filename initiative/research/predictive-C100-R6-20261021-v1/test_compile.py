import ast
import copy
import importlib.util
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / "predictive-R6-additive-20261020-v1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = load("C100_R6_compiler_test", HERE / "compile.py")
F = load("C100_R6_process_fixture", PRIOR / "test_supervisor.py")


@pytest.fixture
def case(tmp_path, monkeypatch):
    supervisor = types.ModuleType("actual_C100_R6_successor")
    exec(
        compile(C.supervisor_source((PRIOR / "supervisor.py").read_text()), "successor", "exec"),
        supervisor.__dict__,
    )
    monkeypatch.setattr(F, "s", supervisor)
    base = F.case.__wrapped__(tmp_path, monkeypatch)
    return base, supervisor


def test_distinct_complete_children_fuse_without_changing_sources(case):
    base, supervisor = case
    c, r = base["sources"]()
    binding = base["bound"](c, r)
    binding["public_watchdog_seconds"] = 900
    result = base["run"](c, r, binding)
    assert result["effective_watchdog_seconds"] == 900
    assert result["total_watchdog_seconds"] == 32400
    assert result["stage_order"] == ["C", "R6"]
    assert result["source_sha256"]["C"] == supervisor.digest(c)
    assert result["source_sha256"]["R6"] == supervisor.digest(r)
    assert (base["root"] / "submission.csv").is_file()


def test_hidden_complete_CSV_not_rejected_only_for_qualification_memory_threshold(case):
    base, s = case
    proof = copy.deepcopy(base["c_proof"])
    proof["models"][0]["memory_accepted"] = False
    binding = base["bound"](*base["sources"]())
    with pytest.raises(ValueError, match="Exact complete trained"):
        s.check_c_receipt(
            proof, binding, roster_sha256=s.PUBLIC_ROSTER, studies=3, csv_bytes=base["c_csv"]
        )
    proof["test_roster_sha256"] = "0" * 64
    s.check_c_receipt(proof, binding, roster_sha256="0" * 64, studies=3, csv_bytes=base["c_csv"])


def test_exact_children_qualification_and_full_model_are_bound(case, monkeypatch):
    base, s = case
    chosen = {"model": {"sha256": "a" * 64}, "architecture": {"context_cap": 24}}
    public = base["c_proof"]
    public["models"][0]["candidate_sha256"] = C.digest(C.canonical(chosen))
    qualification = {
        **public,
        "status": "actual_trained_predictive_public_T4_qualification_accepted",
    }
    qualification["models"][0]["prediction_sha256"] = "c" * 64
    original = dict(
        mode="competition",
        branch="C100",
        qualification=qualification,
        models=[chosen],
        **{
            k: public[k]
            for k in (
                "serving_worker_sha256",
                "serving_contract_sha256",
                "serving_helpers_sha256",
                "model_dataset_ref",
            )
        },
    )
    c = ("BINDING = " + repr(original) + "\n").encode()
    r = b"# synthetic isolated R6 test fixture; no real external source\n"
    for key, value in (
        ("C100", C.digest(c)),
        ("R6", C.digest(r)),
        ("MODEL", "a" * 64),
        ("PUBLIC_CSV", public["CSV_sha256"]),
        ("PUBLIC_ROSTER", public["test_roster_sha256"]),
    ):
        monkeypatch.setattr(C, key, value)
    kwargs = dict(
        supervisor_text=(PRIOR / "supervisor.py").read_text(),
        fusion_text=(PRIOR / "fusion.py").read_text(),
    )
    source, result = C.build(c, r, qualification, public, **kwargs)
    ast.parse(source)
    assert result["binding"]["source_sha256"]["C"] == C.digest(c)
    assert result["binding"]["source_sha256"]["R6"] == C.digest(r)
    assert result["binding"]["fixed_candidate_weight"] == 0.25
    assert result["competition_entry"] is False
    with pytest.raises(ValueError, match="already-executed"):
        C.build(c + b"\n", r, qualification, public, **kwargs)
    bad = copy.deepcopy(public)
    bad["CSV_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="complete public prediction"):
        C.build(c, r, qualification, bad, **kwargs)


@pytest.mark.parametrize(
    "change",
    [
        {"status": None},
        {"publicScore": None},
        {"publicScore": "NaN"},
        {"publicScore": 0.948},
        {"errorDescription": "hidden failed"},
        {"ref": 57482739},
    ],
)
def test_pending_failed_low_or_wrong_model_never_triggers_combined_run(change):
    r = dict(ref=57491555, status="COMPLETE", publicScore=0.949, errorDescription=None)
    assert C.score_trigger(r)
    r.update(change)
    assert not C.score_trigger(r)


def test_public_deadline_expiry_leaves_no_partial_submission(case, monkeypatch):
    base, s = case

    class Timer:
        def __init__(self, delay, callback):
            self.delay, self.callback = delay, callback

        def start(self):
            if self.delay <= 900:
                self.callback()

        def cancel(self):
            pass

    monkeypatch.setattr(s.threading, "Timer", Timer)
    c, r = base["sources"]()
    binding = base["bound"](c, r)
    binding["public_watchdog_seconds"] = 900
    with pytest.raises(TimeoutError):
        base["run"](c, r, binding)
    assert not (base["root"] / "submission.csv").exists()
    assert not (base["root"] / "order").exists()
