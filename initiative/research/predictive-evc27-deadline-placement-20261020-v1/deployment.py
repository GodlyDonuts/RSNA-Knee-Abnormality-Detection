"""Unchanged canonical Kaggle phases after three exact node27 final returns."""

import copy
import json
import os
import time
from pathlib import Path

import contract as d
import owner as fitting

ROOT, HERE = fitting.ROOT, fitting.HERE
CANONICAL = "8fd43b9818c85cbbff9e608ae5ba7d3ae2c6a26f"
CANONICAL_DIR = "initiative/research/predictive-final-deployment-owner-20261020-v1"


def final_reader(source):
    old = 'actual["queue_empty"] is True and actual["node"] == "evc33"'
    d.require(source.count(old) == 1, "Original final acceptance source changed")
    return source.replace(old, 'actual["queue_empty"] is True and actual["node"] == "evc27"')


def finals(original, lineage, directory=HERE):
    directory = Path(directory)
    value = original.full_finals(directory)
    if value is None:
        return None
    transferred = directory / "actual-owner-contexts-all27.json"
    raw = transferred.read_bytes()
    admission = json.loads(raw)
    d.require(admission["status"] == "actual_C0_context_transfer_and_CPU_admission_accepted"
              and admission["admitted_arms"] == ["C", "B", "C0"]
              and set(admission["input_admission_sha256"]) == set(d.ARMS),
              "Actual three-arm shared-cache admission required")
    control = admission["cache_successor_control"]
    d.require(control["source_correction_policy"] == lineage.correction_policy(), "Corrected MRI policy changed")
    lineage.legacy.validate_correction(control, control["accepted_source_correction_proofs"])
    d.require(lineage.accepted_http(control) == control["accepted_http499_proofs"], "HTTP499 lineage differs")
    result = copy.deepcopy(value)
    result["returns"], result["return_receipt_sha256"] = {}, {}
    per_arm = {}
    for arm in d.ARMS:
        f, publication = value["finals"][arm], value["publications"][arm]
        d.require(f["source_revision"] == lineage.BASELINE
                  and f["input_admission_sha256"] == admission["input_admission_sha256"][arm]
                  and f["binding_sha256"] == admission["binding_sha256"][arm],
                  "Final must bind its actual node27 input and initialization")
        path = directory / ("actual-owner-return-" + arm + ".json")
        if not path.exists():
            return None
        returned = json.loads(path.read_bytes())
        d.require(returned["status"] == "actual_predictive_final_export_returned_to33_all_bytes_SHA_verified"
                  and returned["arm"] == arm and returned["source_node"] == "evc27"
                  and returned["destination_node"] == "evc33"
                  and returned["original_training_GPU_job"] == f["GPU_job"]
                  and returned["science_git"] == d.SCIENCE
                  and returned["binding_sha256"] == f["binding_sha256"]
                  and returned["mirror_run_root"] == str(d.return_root(arm))
                  and len(returned["original_six_private_revisions"]) == 6
                  and returned["original_six_private_revisions"][-1] == publication["revision"]
                  and returned["originals_deleted"] is False and returned["new_publication"] is False,
                  "Exact independently verified original arm return required")
        for name, expected in publication["files"].items():
            d.require(returned["objects"]["publication/epoch-006/" + name] == expected,
                      "Returned object differs from original verified publication")
        result["returns"][arm], result["return_receipt_sha256"][arm] = returned, d.sha(path.read_bytes())
        per_arm[arm] = {"input_admission_sha256": f["input_admission_sha256"],
                        "original_admission_receipt_sha256": d.sha(raw), "cache_successor_control": control}
    result["effective_source_lineage"] = {
        "effective_source_policy": lineage.correction_policy(), "per_arm": per_arm,
        "unchanged_worker_export_binds_this_via_input_admission_sha256": True,
        "no_whole_revision_substitution": True}
    return result


def observer_source(source, values, jobs, placement_source):
    """Only physically returned33 bytes retire the original33 model obligations."""
    marker = 'run = Path(f"/tmp/rsna-predictive-full-{arm}-20261020-v2")'
    d.require(source.count(marker) == 1, "Original accounting observer changed")
    roots = {a: str(d.return_root(a)) for a in d.ARMS}
    source = source.replace(marker, f"run = Path({roots!r}[arm])")
    old = 'job["node"] == ("evc27" if arm == "C0" else "evc33")'
    d.require(placement_source.count(old) == 1, "Original paid-log ownership gate changed")
    placement_source = placement_source.replace(old, 'job["node"] == "evc27"')
    checks = "\n    import hashlib\n"
    for arm in d.ARMS:
        path = str(d.return_root(arm) / "return-verified.json")
        checks += (f"    returned = Path({path!r})\n"
                   f"    assert hashlib.sha256(returned.read_bytes()).hexdigest() == {values['return_receipt_sha256'][arm]!r}\n"
                   "    returned_value = json.loads(returned.read_text())\n"
                   f"    assert returned_value['arm'] == {arm!r}\n"
                   "    assert returned_value['source_node'] == 'evc27' and returned_value['destination_node'] == 'evc33'\n")
    d.require(source.count("    paid_runs = {}") == 1, "Original paid accounting changed")
    source = source.replace("    paid_runs = {}", checks +
        f"    paid_logs = _placement.paid_shared_logs({jobs!r})\n    paid_runs = {{}}")
    source = source.replace('"paid_full_run_bytes": paid_runs,',
                            '"paid_full_run_bytes": paid_runs, "paid_original_GPU_logs": paid_logs,')
    return ("import types\n_placement=types.ModuleType('all27_paid_logs')\n"
            f"exec(compile({placement_source!r},'placement.py','exec'),_placement.__dict__)\n" + source)


def rebind_config(value, paid):
    value = copy.deepcopy(value)
    d.require(type(paid["files"]) is int and 0 <= paid["files"] <= 6
              and paid["files"] == len(set(paid["paths"])) and paid["originals_deleted"] is False,
              "Only actually paid original six logs retire file reserve")
    d.require(value["funding"]["other_shared_unpaid_files"] >= 368, "Original shared obligations retained")
    value["funding"]["other_shared_unpaid_files"] -= paid["files"]
    value["funding"]["paid_original_GPU_log_files"] = paid
    value["original_runs"] = {a: {"run_root": str(d.return_root(a)),
        "license_path": str(d.return_root(a) / "publication/epoch-006/source-permission.txt")} for a in d.ARMS}
    return value


def run(args, frozen):
    os.umask(0o077)
    p, lineage = frozen["p"], frozen["lineage"]
    source = fitting.git_file(CANONICAL, CANONICAL_DIR + "/owner.py").decode()
    source = final_reader(lineage.derived_deployment_owner(source))
    original = fitting.module(source, "all27_canonical_owner", ROOT / CANONICAL_DIR / "owner.py")
    decision = json.loads(args.deployment_decision.read_bytes())
    original.validate_decision(decision)
    lineage.deployment_funding(decision)
    d.require(decision["root"] == "/tmp/rsna-predictive-final-deployment-20261020-v1-all27"
              and decision["portable_placement_source_git"] == args.source_git
              and decision["placement_handoff_sha256"] == p.file_sha(args.handoff)
              and decision["deployment_handoff_sha256"] == p.file_sha(args.deployment_handoff)
              and decision["runtime"]["public_VM_startup_allowance_seconds"] >= 120,
              "Distinct account root/source and original3360-second reserve required")
    population = decision["official_population_receipt"]
    body = fitting.git_file(CANONICAL, population["path"])
    d.require(d.sha(body) == population["sha256"] and json.loads(body)["test_studies"]["approximate_count"]
              == decision["runtime"]["planning_study_count"], "Official population evidence changed")
    journal = ROOT / "artifacts" / (HERE.name + "-deployment")
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    p.once(journal / "owner-intent.json", {"pid": os.getpid(), "placement_source_git": args.source_git,
        "canonical_account_control_source_git": CANONICAL, "decision_sha256": p.file_sha(args.deployment_decision),
        "deployment_handoff_sha256": p.file_sha(args.deployment_handoff), "automatic_retry": False})
    read_final = original.full_finals
    reader = type("OriginalFinalReader", (), {"full_finals": staticmethod(read_final)})
    original.full_finals = lambda: finals(reader, lineage)
    base = original.Owner

    class All27Deployment(base):
        def __init__(self):
            super().__init__(CANONICAL, decision, args.socket, journal)

        def _observe(self, dataset_slug=None):
            values = finals(reader, lineage)
            d.require(values is not None, "All original returns required before account access")
            receipt = json.loads((HERE / "actual-known-dispatches-all27.json").read_bytes())
            jobs = receipt["jobs"]
            d.require(set(jobs) == set(d.ARMS) and all(jobs[a]["GPU_job"] == values["finals"][a]["GPU_job"]
                                                      for a in d.ARMS), "Actual dispatch/final identities differ")
            self.observer = observer_source(
                fitting.git_file(CANONICAL, CANONICAL_DIR + "/observe.py.txt").decode(), values, jobs,
                frozen["originals"]["placement.py"])
            return super()._observe(dataset_slug)

        def config(self, observed, mode):
            return rebind_config(super().config(observed, mode), observed["paid_original_GPU_logs"])

    d.require(time.time() < original.LAUNCH_END, "No new deployment beyond latest entry target")
    try:
        All27Deployment().run()
    except BaseException as error:
        p.once(journal / "stopped.json", {"error_type": type(error).__name__, "automatic_retry": False,
                                          "preserve_original_phases_and_qualified_baseline": True})
        raise
