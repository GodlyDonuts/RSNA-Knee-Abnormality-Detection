"""Local operational tests: frozen dispatch graph and one-copy arm admission."""

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PORTABLE = REPO / "initiative/research/predictive-c0-portable-placement-20261020-v1"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


m = load(HERE / "remote.py", "deadline_remote_under_test")


@pytest.fixture(autouse=True)
def private_files():
    previous = os.umask(0o077)
    yield
    os.umask(previous)


def decision():
    return {"node": "evc27", "dispatch_order": list(m.ARMS),
            "scientific_source_git": m.SCIENCE, "original_preparation_CPU_job": "979667",
            "funding": copy.deepcopy(m.FUNDING)}


def inherited(monkeypatch):
    p = load(PORTABLE / "placement.py", "original_portable_placement")
    monkeypatch.setitem(sys.modules, "placement", p)
    source = (PORTABLE / "remote.py").read_text()
    n = {"__name__": "original_remote", "__portable_source__": source}
    exec(compile(source, "frozen_portable", "exec"), n)
    v = {"deadline_decision": decision(), "deadline_code_sha256": {"remote.py": "d" * 64},
         "shared_bytes": 123 * m.MIB, "shared_files": 376,
         "original_remote": (REPO / "initiative/research/predictive-full-fitting-20261020-v2/owner_remote.py.txt").read_text()}
    m.install(n, v)
    return n, p, v


def test_exact_funded_total_and_no_optional_credit(monkeypatch):
    _, p, _ = inherited(monkeypatch)
    expected = 394815078400
    assert p.capacity(expected + 1)["required_bytes"] == expected
    with pytest.raises(ValueError, match="not funded"):
        p.capacity(expected)
    for key in ("generative_pilot_bytes", "descriptor_bytes", "additional_control_bytes"):
        d = decision()
        d["funding"][key] = 0
        with pytest.raises(ValueError, match="reserved"):
            m.validate_decision(d)


def test_actual_frozen_control_dispatches_all27_in_priority_order(monkeypatch, tmp_path):
    n, p, v = inherited(monkeypatch)
    p.ROOT27 = tmp_path
    ns = n["namespace"](v, "evc27")
    assert ns["ARMS"] == ("C", "B", "C0")
    assert ns["ROOT"] == tmp_path
    assert ns["PYTHON"] == p.PY27
    ns["SCIENCE"] = m.SCIENCE
    ns["preparation"] = lambda _: None
    ns["check_node"] = lambda: "a" * 64
    # Preserve the actual original run-root literal in sealed job metadata; the
    # dispatch only checks absence and never creates these paths in this test.
    ns["base_other"] = lambda _: 99
    ns["runpy"] = SimpleNamespace(run_path=lambda *a, **k: {
        "check_space": lambda *a, **k: {"fresh": True}})
    admission = tmp_path / "full-admission"
    admission.mkdir()
    hashes = {}
    for arm in m.ARMS:
        path = admission / f"binding-{arm}.private.json"
        path.write_text(json.dumps({"arm": arm}))
        hashes[arm] = ns["sha"](path)
    actual = {"arms": list(m.ARMS), "binding_sha256": hashes}
    (admission / "actual.json").write_text(json.dumps(actual))
    calls = []

    def submit(argv, **kwargs):
        script = Path(argv[-1]).read_text()
        assert "--nodelist=evc27" in script and "--nodelist=evc33" not in script
        assert "--gres=gpu:nvidia_h100_pcie:1" in script and "--time=08:00:00" in script
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout=str(900 + len(calls)), stderr="")

    ns["subprocess"] = SimpleNamespace(run=submit)
    monkeypatch.setenv("SLURM_JOB_ID", "700")
    result = ns["dispatch"]({**v, "admitted": actual, "owner_git": "1" * 40})
    assert list(result["jobs"]) == list(m.ARMS)
    assert all(j["node"] == "evc27" for j in result["jobs"].values())
    assert result["submitted_arm_count"] == 3
    assert result["placement_total_GPU_limit"] == 2
    checked = m.reconcile_dispatch(p, {"owner_git": "1" * 40,
                                       "portable": {"binding_sha256": hashes}})
    assert checked["submission_completeness_known"] is True
    assert checked["jobs"] == result["jobs"]
    (tmp_path / "dispatch/submitted.json").unlink()
    (tmp_path / "dispatch/B-submitted.json").unlink()
    checked = m.reconcile_dispatch(p, {"owner_git": "1" * 40,
                                       "portable": {"binding_sha256": hashes}})
    assert checked["submission_completeness_known"] is False
    assert set(checked["jobs"]) == {"C", "C0"}
    assert checked["unsealed_arms"] == ["B"]
    # An apparently successful sbatch response alone is deliberately not adopted.
    assert (tmp_path / "dispatch/B-sbatch.json").exists()
    with pytest.raises(AssertionError):
        ns["dispatch"]({**v, "admitted": actual, "owner_git": "1" * 40})
    assert len(calls) == 3


def test_source_node_cannot_dispatch(monkeypatch):
    n, _, v = inherited(monkeypatch)
    with pytest.raises(ValueError, match="CPU validation only"):
        n["main"]({**v, "mode": "dispatch", "node": "evc33"})


def test_only_actual_node27_job_logs_can_receive_paid_credit(monkeypatch, tmp_path):
    _, p, _ = inherited(monkeypatch)
    jobs = {}
    for i, arm in enumerate(m.ARMS):
        job = str(i + 123)
        name = "rsna-predictive-full2-" + arm
        jobs[arm] = {"GPU_job": job, "arm": arm, "source_git": m.SCIENCE,
                     "node": "evc27", "job_name": name}
        for field, suffix in (("stdout", "out"), ("stderr", "err")):
            path = tmp_path / f"{name}-{job}.{suffix}"
            path.write_text("bounded telemetry\n")
            path.chmod(0o600)
            jobs[arm][field] = str(path)
    assert p.paid_shared_logs(jobs, tmp_path)["files"] == 6
    jobs["B"]["node"] = "evc33"
    with pytest.raises(ValueError, match="all27"):
        p.paid_shared_logs(jobs, tmp_path)


def test_admission_reuses_one_index_but_preserves_each_actual_recipe(monkeypatch, tmp_path):
    n, p, v = inherited(monkeypatch)
    p.ROOT27 = tmp_path
    out = tmp_path / "full-admission"
    out.mkdir()
    master = json.loads((REPO / "initiative/research/predictive-full-fitting-20261020-v1/engineering-decision.json").read_text())
    (tmp_path / "master.json").write_text(json.dumps(master))
    all_same = {"context_index": {"path": "/one/context-index", "sha256": "a" * 64},
                "parent_weight": {"path": "/same/parent", "sha256": "b" * 64},
                "targets": {"path": "/same/targets", "sha256": "c" * 64}}
    input_body = {"cache_successor_control": {"exact_corrected_proofs": "preserved"},
                  "source_context_index_sha256": "x" * 64}

    def original(value, capsule, rows, transferred):
        for name, obj in {
            "engineering-C0.json": {**master, "arm": "C0", "recipe": master["recipes"]["C0"]},
            "input-C0.json": input_body,
            "funding-C0.json": {"arm": "C0", "other_unpaid_node_liabilities_bytes":
                                 m.ORIGINAL_OTHER + m.OPTIONAL + m.EXTRA},
        }.items():
            p.once(out / name, obj)
        binding = {**all_same, "node": "evc27", "arm": "C0", "source_git": m.SCIENCE,
                   "expected_initial_shared_tensor_sha256": p.INITIAL,
                   "engineering_decision": p.reference(out / "engineering-C0.json"),
                   "input_admission": p.reference(out / "input-C0.json"),
                   "funding": p.reference(out / "funding-C0.json")}
        p.once(out / "binding-C0.private.json", binding)
        result = {"binding_sha256": {"C0": p.file_sha(out / "binding-C0.private.json")}}
        p.once(tmp_path / "portable-accepted.json", result)
        return result

    actual = m.admit_remaining(n, v, original, {}, [], 123)
    assert actual["admitted_arms"] == list(m.ARMS)
    assert actual["additional_context_copies"] == 0
    assert set(actual["input_admission_sha256"]) == set(m.ARMS)
    for arm in m.ARMS:
        binding = json.loads((out / f"binding-{arm}.private.json").read_text())
        assert all(binding[k] == x for k, x in all_same.items())
        assert json.loads(p.read_ref(binding["engineering_decision"]))["recipe"] == master["recipes"][arm]
        assert json.loads(p.read_ref(binding["input_admission"])) == input_body
        assert binding["arm"] == arm
    assert json.loads((tmp_path / "portable-accepted.json").read_text()) == actual


@pytest.mark.parametrize("arm", m.ARMS)
def test_return_specialization_keeps_transports_and_selects_exact_arm(monkeypatch, arm):
    n, _, _ = inherited(monkeypatch)
    send, receive = m.specialize_returns(n["__portable_source__"], arm, n)
    assert send.__name__ == "return_sender" and receive.__name__ == "return_receiver"
    assert f"full-admission/binding-{arm}.private.json" in send.__code__.co_consts
    assert arm in send.__code__.co_consts
    assert "actual_predictive_final_export_returned_to33_all_bytes_SHA_verified" in receive.__code__.co_consts
    assert m.return_root(arm).name == f"rsna-predictive-{arm}-evc27-return-20261020-v1"
