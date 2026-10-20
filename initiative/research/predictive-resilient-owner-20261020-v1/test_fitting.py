import importlib.util
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).parent
POLICY = HERE.parent / "predictive-deadline-admission-20261020-v1"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


fitting = load("resilient_fitting_test", HERE / "fitting.py")
deadline = load("resilient_deadline_test", POLICY / "policy.py")
placement = load("resilient_placement_test", POLICY / "placement_failure.py")


def timestamp(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def once(path, value):
    with path.open("x") as handle:
        json.dump(value, handle)


def ready(progress, published):
    candidates = [
        (epoch, arm) for arm, item in progress["arms"].items()
        for epoch in range(1, item["sealed_epochs"] + 1) if (arm, epoch) not in published
    ]
    if candidates:
        epoch, arm = min(candidates)
        return arm, epoch
    return None


class Backend:
    def __init__(self):
        self.calls, self.publications = [], []
        self.transfer_gate = threading.Event()
        self.transfer_started = threading.Event()
        self.c0_block = False
        self.c0_failure = False
        self.c0_dispatch_failure = False
        self.c0_bad_dispatch_receipt = False
        self.publish_failure = False
        self.c0_observation_failure = False
        self.c0_observation_failures_left = 0
        self.return_failure = False
        self.now = timestamp("2026-10-20T16:00:00")
        self.advance_on_admit = None
        self.advance_on_transfer = None
        self.complete = True
        self.release_transfer_on_B_publication = True
        self.publish_lock = threading.Lock()

    def sleep(self, duration):
        self.now += duration
        time.sleep(0.001)


class FakeOwner:
    def __init__(self, root, backend):
        self.backend = backend
        self.journal = root / "original-phase-journal"
        self.journal.mkdir()
        once(self.journal / "owner-intent.json", {"original": True})
        self.audit = root / "original-phase-audit"
        self.audit.mkdir()
        self.args = SimpleNamespace(source_git="2b834507e19c405971228e1bc405462cd370f560",
                                    poll_seconds=1)
        self.payload = {"prepared": {"source_git": fitting.SCIENCE}, "unmodified": {"items": []}}
        self.deadline = timestamp("2026-10-22T09:00:00")
        self.original = SimpleNamespace(
            cache_receipts=lambda: {"raw": {"original": "raw"}, "context": {"original": "context"}},
            completed=lambda state: state["state"] == "COMPLETED",
            terminal=lambda state: state["state"] in {"COMPLETED", "FAILED", "TIMEOUT"},
            ready_epoch=ready,
        )

    def invoke(self, node, mode, phase=None, extra=None):
        b = self.backend
        b.calls.append((node, mode, phase))
        # If this mutation reaches another task's payload, the isolation test fails.
        self.payload["unmodified"]["items"].append(mode)
        if mode == "admit":
            if b.advance_on_admit is not None:
                b.now = b.advance_on_admit
            return {"arms": ["B", "C"]}
        if mode == "dispatch":
            arms = ("C0",) if node == "evc27" else ("B", "C")
            if node == "evc27":
                assert self.payload["portable"]["status"] == "transfer_verified"
                if b.c0_dispatch_failure:
                    raise ConnectionError("Submission may exist; original intent must be reconciled")
            value = {
                "status": "actual_predictive_placement_jobs_submitted_once",
                "source_git": fitting.SCIENCE,
                "jobs": {arm: {"GPU_job": {"B": "100", "C": "101", "C0": "102"}[arm],
                               "arm": arm, "node": node, "source_git": fitting.SCIENCE}
                         for arm in arms},
            }
            if node == "evc27" and b.c0_bad_dispatch_receipt:
                value["jobs"]["C0"]["node"] = "evc33"
            once(self.journal / (phase + "-result.json"), value)
            once(self.audit / ("actual-owner-" + phase + ".json"), value)
            return value
        if mode == "progress":
            assert (self.journal / ("known-dispatches-" + ("C0" if node == "evc27" else "BC") +
                                   ".json")).is_file()
            if node == "evc27" and (b.c0_observation_failure or b.c0_observation_failures_left):
                b.c0_observation_failures_left = max(0, b.c0_observation_failures_left - 1)
                if b.c0_observation_failure and len(b.publications) == 12:
                    b.now = fitting.PRESERVATION_END
                raise ValueError("Original C0 progress parser refused a row")
            return {"arms": {
                arm: {"GPU_job": job["GPU_job"],
                      "slurm": {"state": "COMPLETED" if b.complete else "RUNNING"},
                      "terminal": {"arm": arm, "GPU_job": job["GPU_job"]} if b.complete else None,
                      "stderr_bytes": 0, "sealed_epochs": 6 if b.complete else 1}
                for arm, job in extra["dispatched"]["jobs"].items()
            }}
        if mode == "publish":
            assert b.publish_lock.acquire(blocking=False), "Publications must be serialized"
            try:
                b.publications.append((extra["arm"], extra["epoch"]))
                if b.release_transfer_on_B_publication and extra["arm"] == "B":
                    b.transfer_gate.set()
                if b.publish_failure:
                    raise TimeoutError("Original publication outcome unresolved")
                return {"status": fitting.VERIFIED, "objects": 5, "arm": extra["arm"],
                        "epoch": extra["epoch"], "source_git": fitting.SCIENCE}
            finally:
                b.publish_lock.release()
        raise AssertionError(mode)

    def transfer(self, returning=False):
        b = self.backend
        b.calls.append(("evc27" if returning else "evc33", "transfer", returning))
        assert set(self.payload["all_dispatched_jobs"]) >= {"B", "C"}
        if returning:
            assert len([p for p in b.publications if p[0] == "C0"]) == 6
            if b.return_failure:
                raise ConnectionError("Original return unresolved")
            return {"status": "final_return_verified"}
        b.transfer_started.set()
        if b.c0_block:
            assert b.transfer_gate.wait(2), "B/C did not publish while C0 transfer blocked"
        if b.c0_failure:
            raise ConnectionError("Original transfer unresolved")
        if b.advance_on_transfer is not None:
            b.now = b.advance_on_transfer
        return {"status": "transfer_verified"}


def run(owner):
    return fitting.run(
        owner, deadline_policy=deadline, placement_policy=placement, once=once,
        audit_dir=owner.audit, clock=lambda: owner.backend.now, sleep=owner.backend.sleep,
    )


def test_B_C_publication_continues_while_original_C0_transfer_blocks(tmp_path):
    backend = Backend()
    backend.c0_block = True
    owner = FakeOwner(tmp_path, backend)
    result = run(owner)
    assert result["complete"] == ["B", "C", "C0"]
    assert result["failed"] == []
    assert len(result["privately_verified_epochs"]) == 18
    assert len(backend.publications) == len(set(backend.publications)) == 18
    assert result["C0_return"] == {"status": "final_return_verified"}
    assert backend.calls.index(("evc33", "publish", "publish-B-001")) < backend.calls.index(
        ("evc27", "dispatch", "dispatch-C0"))
    assert owner.payload["unmodified"] == {"items": []}
    assert set(owner.payload["all_dispatched_jobs"]) == {"B", "C", "C0"}


def test_C0_transfer_failure_keeps_B_C_preservation_and_never_fakes_GPU_failure(tmp_path):
    backend = Backend()
    backend.c0_failure = True
    result = run(FakeOwner(tmp_path, backend))
    assert result["complete"] == ["B", "C"]
    assert result["failed"] == []
    assert set(result["jobs"]) == {"B", "C"}
    assert len(result["privately_verified_epochs"]) == 12
    assert result["C0_placement"]["C0_operation_state"] == "requires_read_only_reconciliation"
    assert not result["C0_placement"]["C0_model_or_Slurm_failure_inferred"]
    assert not any(node == "evc27" and mode == "dispatch" for node, mode, _ in backend.calls)
    assert len([x for x in backend.calls if x[1] == "transfer"]) == 1


def test_C0_dispatch_uncertainty_does_not_invent_or_retry_an_unknown_job(tmp_path):
    backend = Backend()
    backend.c0_dispatch_failure = True
    result = run(FakeOwner(tmp_path, backend))
    assert result["complete"] == ["B", "C"] and result["failed"] == []
    assert set(result["jobs"]) == {"B", "C"}
    assert result["C0_placement"]["phase"] == "dispatch-C0"
    assert not result["C0_placement"]["C0_GPU_job_known"]
    assert len([x for x in backend.calls if x == ("evc27", "dispatch", "dispatch-C0")]) == 1


def test_invalid_C0_dispatch_receipt_is_unresolved_without_stopping_B_C(tmp_path):
    backend = Backend()
    backend.c0_bad_dispatch_receipt = True
    result = run(FakeOwner(tmp_path, backend))
    assert result["complete"] == ["B", "C"] and result["failed"] == []
    assert set(result["jobs"]) == {"B", "C"}
    assert len(result["privately_verified_epochs"]) == 12
    assert result["C0_placement"]["phase"] == "dispatch-C0"
    assert not result["C0_placement"]["C0_GPU_job_known"]


def test_transfer_crossing_GPU_cutoff_still_preserves_already_submitted_B_C(tmp_path):
    backend = Backend()
    backend.advance_on_transfer = timestamp("2026-10-21T22:00:01")
    result = run(FakeOwner(tmp_path, backend))
    assert result["complete"] == ["B", "C"] and len(result["privately_verified_epochs"]) == 12
    assert result["C0_placement"]["phase"] == "dispatch-C0"
    assert result["C0_placement"]["reason"] == "launch_cutoff"
    assert not any(node == "evc27" and mode == "dispatch" for node, mode, _ in backend.calls)


def test_after_cache_admission_cutoff_no_phase_or_GPU_is_started(tmp_path):
    backend = Backend()
    backend.now = timestamp("2026-10-21T17:15:01")
    result = run(FakeOwner(tmp_path, backend))
    assert result["reason"] == "no_new_admission_after_dispatch_cutoff"
    assert backend.calls == [] and result["jobs"] == {}


def test_admission_crossing_dispatch_cutoff_does_not_submit_GPU(tmp_path):
    backend = Backend()
    backend.advance_on_admit = timestamp("2026-10-21T22:00:01")
    result = run(FakeOwner(tmp_path, backend))
    assert result["reason"] == "no_new_B_C_dispatch_after_cutoff"
    assert backend.calls == [("evc33", "admit", "admission-BC")]


def test_uncertain_publication_is_never_retried_but_known_jobs_remain_observed(tmp_path):
    backend = Backend()
    backend.publish_failure = True
    result = run(FakeOwner(tmp_path, backend))
    assert len(backend.publications) == 1
    assert result["publication_problem"]["further_publication_mutations_stopped"]
    assert result["complete"] == ["B", "C", "C0"]
    assert result["C0_return"] is None
    assert result["remote_jobs_cancelled"] == []


def test_C0_observation_failure_does_not_stop_B_C_publication_or_fake_terminal(tmp_path):
    backend = Backend()
    backend.c0_observation_failure = True
    owner = FakeOwner(tmp_path, backend)
    result = run(owner)
    assert result["complete"] == ["B", "C"] and result["failed"] == []
    assert len(result["privately_verified_epochs"]) == 12
    assert set(result["jobs"]) == {"B", "C", "C0"}
    assert result["reason"] == "preservation_deadline_existing_jobs_require_read_only_continuation"
    assert result["unresolved_node_observations"]["evc27"]["error_type"] == "ValueError"
    assert len(list(owner.audit.glob("actual-observation-unresolved-evc27-*.json"))) == 1
    assert not result["unresolved_node_observations"]["evc27"]["training_failure_inferred"]


def test_C0_read_only_observation_can_recover_without_repeating_dispatch(tmp_path):
    backend = Backend()
    backend.c0_observation_failures_left = 1
    owner = FakeOwner(tmp_path, backend)
    result = run(owner)
    assert result["complete"] == ["B", "C", "C0"]
    assert result["unresolved_node_observations"] == {}
    assert len(result["privately_verified_epochs"]) == 18
    assert (owner.audit / "actual-observation-recovered-evc27-001.json").is_file()
    assert len([x for x in backend.calls if x == ("evc27", "dispatch", "dispatch-C0")]) == 1


def test_final_return_failure_is_separate_from_C0_model_success(tmp_path):
    backend = Backend()
    backend.return_failure = True
    result = run(FakeOwner(tmp_path, backend))
    assert result["complete"] == ["B", "C", "C0"]
    assert result["failed"] == []
    assert result["reason"] == "original_C0_final_return_unresolved_no_retry"
    assert len([x for x in backend.calls if x[1:] == ("transfer", True)]) == 1


def test_expired_preservation_owner_does_not_start_any_operation(tmp_path):
    backend = Backend()
    backend.now = fitting.PRESERVATION_END
    with pytest.raises(ValueError, match="preservation deadline"):
        run(FakeOwner(tmp_path, backend))
    assert backend.calls == []


@pytest.mark.parametrize("name", [
    "dispatch-BC-intent.json", "dispatch-C0-result.json", "contexts-C0-intent.json",
    "publish-B-001-transport.json", "deadline-cache_admission.json", "owner-ended.json",
])
def test_prephase_only_refuses_any_prior_operation_evidence(tmp_path, name):
    backend = Backend()
    owner = FakeOwner(tmp_path, backend)
    once(owner.journal / name, {})
    with pytest.raises(ValueError, match="Prephase-only"):
        run(owner)
    assert backend.calls == []
