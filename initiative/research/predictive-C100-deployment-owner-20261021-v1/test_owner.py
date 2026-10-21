import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("C100_owner_tests", HERE / "owner.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def valid():
    return dict(
        full_scheduler=dict(
            job_id="995351", state="COMPLETED", exit_code="0:0", queue_empty=True, node="evc27"
        ),
        terminal=dict(
            GPU_job="995351",
            source_git=M.SOURCE,
            binding_sha256=M.BINDING,
            branch="C100",
            epochs=6,
            updates=2916,
        ),
        publications={
            str(i): dict(
                epoch=i,
                GPU_job="995351",
                source_git=M.SOURCE,
                objects=5,
                status="actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified",
            )
            for i in range(1, 7)
        },
    )


def test_complete_original_only():
    assert len(M.final_evidence(valid())["epoch_publications"]) == 6


@pytest.mark.parametrize(
    "change",
    [
        "running",
        "failed",
        "wrong_job",
        "wrong_source",
        "wrong_binding",
        "partial",
        "missing_publication",
        "uncertain_publication",
    ],
)
def test_reject_partial_or_foreign_original(change):
    v = valid()
    if change == "running":
        v["full_scheduler"]["state"] = "RUNNING"
    if change == "failed":
        v["full_scheduler"]["exit_code"] = "1:0"
    if change == "wrong_job":
        v["terminal"]["GPU_job"] = "981620"
    if change == "wrong_source":
        v["terminal"]["source_git"] = "0" * 40
    if change == "wrong_binding":
        v["terminal"]["binding_sha256"] = "0" * 64
    if change == "partial":
        v["terminal"]["updates"] = 2915
    if change == "missing_publication":
        del v["publications"]["6"]
    if change == "uncertain_publication":
        v["publications"]["6"]["status"] = "intent"
    with pytest.raises(ValueError):
        M.final_evidence(v)


def test_actual_pilot_serialization_bound():
    pilot = {"original": True}
    value = dict(
        pilot=pilot,
        pilot_sha256=M.sha((json.dumps(pilot, sort_keys=True) + "\n").encode()),
        pilot_scheduler={},
    )
    frozen = ({"code_sha256": {"publish_epoch.py": "a" * 64}},)
    assert M.options(value, frozen)["publisher_sha256"] == "a" * 64
    value["pilot"]["original"] = False
    with pytest.raises(ValueError):
        M.options(value, frozen)


def test_original_phase_actions_keep_uncertain_mutation_fail_closed(monkeypatch, tmp_path):
    b = M.original()
    o = object.__new__(b.Owner)
    o.journal, o.pin, o.socket, o.counter = tmp_path, "a" * 40, "unused", 0
    o.launcher = tmp_path / "unused.py"
    o.observe = lambda **_: dict(canonical_claim_present=False, allowance=2)
    o.config = lambda *_: {}
    o.sleep = lambda *_: pytest.fail("An uncertain mutation must never retry")
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1, stdout="unknown transport outcome", stderr="")

    monkeypatch.setattr(b.subprocess, "run", fail)
    with pytest.raises(AssertionError, match="Uncertain"):
        o.action("submit", "C")
    assert len(calls) == 1 and (tmp_path / "submit-C-intent.json").exists()
    with pytest.raises(AssertionError, match="Existing phase intent"):
        o.action("submit", "C")
    assert len(calls) == 1


def test_phase_mapping_and_single_model_budget():
    b = M.original()
    assert b.ARMS == ("C",) and b.DEPLOY == M.DEPLOY
    assert (
        b.phase_result(
            "prepare",
            "qualification",
            {"status": "actual_single_C100_final_export_prepared_CPU_only"},
        )
        == "done"
    )
    with pytest.raises(AssertionError):
        b.phase_result(
            "prepare",
            "qualification",
            {"status": "actual_three_predictive_final_exports_prepared_CPU_only"},
        )
    assert (
        b.phase_result("submit", "C", {"status": "waiting_existing_canonical_claim_no_mutation"})
        == "wait"
    )


def test_shared_claim_scope_and_fresh_floor(monkeypatch):
    b = M.original()
    policy = b.load_frozen(
        M.PIN,
        M.ROOT / "initiative/research/predictive-final-deployment-20261020-v1/policy.py",
        "owner_funding_test",
    )
    decision = json.loads((HERE / "decision.json").read_text())
    o = SimpleNamespace(
        base=b,
        decision=decision,
        qualification=None,
        remaining_public=1,
        contract_options={},
        evidence={},
    )
    obs = dict(
        observed_at=b.time.time(),
        paid_original_GPU_logs=dict(files=6, paths=list("abcdef"), originals_deleted=False),
        node_free_bytes=600 * 2**30,
        shared_free_bytes=22 * 2**30,
        shared_free_files=5373,
        original_license="/tmp/license",
        original_parent_source="/tmp/parent",
    )
    cfg = M.config(o, obs, "qualification", policy)
    assert cfg["funding"]["other_shared_unpaid_files"] + cfg["funding"]["new_shared_files"] == 370
    assert cfg["funding"]["other_node_unpaid_bytes"] == 512 * 2**30
    bad = copy.deepcopy(obs)
    bad["shared_free_files"] = 5369
    with pytest.raises(ValueError, match="reserve"):
        M.config(o, bad, "qualification", policy)
    bad = copy.deepcopy(obs)
    bad["observed_at"] -= 181
    with pytest.raises(ValueError, match="Fresh"):
        M.config(o, bad, "qualification", policy)


def test_observer_uses_actual_owning_node_and_exact_log_projection():
    c = M.training()
    f = c.frozen(SimpleNamespace(source_git=M.SOURCE))
    body = M.observer_body(f, c.original_jobs())
    compile(body, "observer", "exec")
    assert 'value["node"] == "evc27"' in body
    assert 'job["node"] == "evc27"' in body
    assert "bound['license']['path']" in body
