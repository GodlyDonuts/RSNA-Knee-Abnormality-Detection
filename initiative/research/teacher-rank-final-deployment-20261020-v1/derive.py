"""Narrow role/provenance delta over the immutable, already launched pointwise route."""

import ast
import hashlib
import json
import subprocess

PARENT = "01cbe205080487b3fb39fbf7c0d28e112fcd7d77"
P = "initiative/research/pointwise-final-deployment-20261020-v1/"
ORIGINAL_PREFIX = "initiative/research/S6-Kaggle-production-route-20261019-v1/"
OLD_MODEL = "da61c53480931166cd13c89a30f8062e386353e77c257fc9a94dd26afb215084"
MODEL = "d478a502e6fd33be2ad9d781acd91d3d2b6b252629262031fba8282a91cf5bca"
NAME = "teacher-rank-final-deployment-20261020-v1"


def replace(source, before, after):
    assert source.count(before) >= 1, "Expected immutable parent site missing"
    return source.replace(before, after)


def parent_files(repo, manifest):
    assert manifest["source_git"] == PARENT
    result = {}
    for path, expected in manifest["files"].items():
        text = subprocess.check_output(["git", "show", PARENT + ":" + path], cwd=repo)
        assert hashlib.sha256(text).hexdigest() == expected
        result[path] = text.decode()
    return result


def accepted(config, decision_text):
    plan = config["runtime_plan"]
    assert plan["status"] == "accepted_teacher_rank_final_runtime_budget", (
        "Actual pointwise timing decision still required; no phase intent"
    )
    assert (
        plan["actual_same_graph_timing_receipt_sha256"]
        == hashlib.sha256(decision_text.encode()).hexdigest()
    )
    decision = json.loads(decision_text)
    assert decision["status"] == "accepted_teacher_rank_final_runtime_and_storage_plan"
    assert decision["actual_pointwise_qualification_passed"] is True
    assert len(decision["actual_pointwise_qualification_receipt_sha256"]) == 64
    assert decision["qualification_cap_seconds"] == plan["qualification_cap_seconds"]
    assert decision["public_watchdog_seconds"] == plan["public_watchdog_seconds"]
    assert (
        type(plan["qualification_cap_seconds"]) is int
        and 0 < plan["qualification_cap_seconds"] <= 600
    )
    assert (
        type(plan["public_watchdog_seconds"]) is int and 0 < plan["public_watchdog_seconds"] <= 300
    )
    assert plan["predictive_reserved_VM_seconds"] >= 3000
    assert plan["same32_case_checks_and_inference_unchanged"] is True
    return plan


def sources(parent, config, decision_text):
    plan = accepted(config, decision_text)
    adapter = {"__name__": "frozen_launched_pointwise_adapter"}
    exec(parent[P + "adapter.py"], adapter)
    function_text = parent[P + "policy_functions.py"]
    functions = {
        n.name: ast.get_source_segment(function_text, n)
        for n in ast.parse(function_text).body
        if isinstance(n, ast.FunctionDef)
    }
    policy = adapter["policy"](parent[ORIGINAL_PREFIX + "serving_policy.py.txt"], functions)
    program = adapter["program"](
        parent[ORIGINAL_PREFIX + "serving_cloud.py.txt"], parent[P + "helpers.py"]
    )
    # This global role delta affects operational symbols/status only, never compiled model bodies.
    program = replace(program, "rsna-pointwise-final-deployment-20261020-v1", "rsna-" + NAME)
    program = replace(program, "pointwise", "teacher_rank")
    policy = replace(policy, "pointwise", "teacher_rank")
    program = replace(program, "rsna-teacher_rank-final-e3-", "rsna-teacher-rank-final-e3-")
    policy = replace(policy, "rsna-teacher_rank-e3-", "rsna-teacher-rank-e3-")
    for old, new in {
        OLD_MODEL: MODEL,
        "977528": "979767",
        "a0851fc19904bc6b86fd93026f86f5e1b86746d3": "4a441fffd22985023dc712401b0880ee7dc9f6fd",
        "e36bb88ffa2aa4ac62680a29a5d7ba0fdd7e7bf6": "60a64c2be5d59f60e458caca1b6f475f82a95fa7",
        "c398324df327fdfdc541f1db56857764dc823029d5719510dc69d88532a60637": "69f0d91347a49fde17b4293b3f21493a41bdd8f176aa6d77156540a04a161556",
        "5d04caf90941aed231e38838528c3fd8cecb626214fd6e81b645810d1b0a2f10": "dd3b7ea13383c5132a3d02c60e87aa65767a21efe8032de43c2bd524718f455e",
    }.items():
        program = replace(program, old, new)
    program = replace(program, ">= 121 * 2**20", ">= 122 * 2**20")
    original_scope = (
        '    used = sum(p.stat().st_size for p in D.rglob("*") if p.is_file()) if D.exists() else 0'
    )
    extra_scope = """    pointwise_scope = Path("/tmp/rsna-pointwise-final-deployment-20261020-v1")
    pointwise_paid = sum(p.stat().st_size for p in pointwise_scope.rglob("*")
                         if p.is_file() and not p.is_symlink()) if pointwise_scope.exists() else 0
    assert pointwise_paid <= 640 * 2**20
    other_node_unpaid += max(0, 640 * 2**20 - pointwise_paid)
"""
    program = replace(program, original_scope, extra_scope + original_scope)
    status_call = '    state = remote.call("datasets.DatasetApiService", "GetDatasetStatus", {"ownerSlug": owner, "datasetSlug": name})'
    # Parent formatting is frozen; replace the actual statement found by AST, not guesses.
    helper = next(
        n
        for n in ast.parse(program).body
        if isinstance(n, ast.FunctionDef) and n.name == "dataset_phase"
    )
    status = next(
        n
        for n in helper.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "state" for t in n.targets)
    )
    status_call = ast.get_source_segment(program, status)
    adapted = (
        """try:
        """
        + status_call
        + """
    except w.RemoteError as error:
        if str(error) != "Kaggle GetDatasetStatus HTTP 404":
            raise
        assert row["state"] == "teacher_rank_dataset_create_returned"
        intent = read(D / "dataset-intent.json")
        assert intent["slug"] == row["slug"] == DATASET_SLUG
        assert intent["model_sha256"] == row["model_sha256"] == MODEL_SHA
        assert intent["version"] == row["version"] == 1 and intent["automatic_retry"] is False
        response = read(D / "dataset-response.json")
        assert response["returned"] is True and response["final_create_rpc_calls"] == 1
        assert response["status"] == "Ok" and response["private_requested"] is True
        try:
            identity = remote.dataset(DATASET_SLUG)
        except w.RemoteError as lookup_error:
            if str(lookup_error) != "Kaggle GetDataset HTTP 404":
                raise
            return {"status": "waiting_original_dataset_ingestion_no_mutation",
                    "actual_status": "ORIGINAL_RETURNED_CREATE_NOT_YET_METADATA_VISIBLE",
                    "identity_verified": False, "new_Dataset_calls": 0,
                    "bounded_by_finite_owner_seconds": 12 * 3600}

        assert identity.get("ref") == DATASET_SLUG and type(identity.get("id")) is int and identity["id"] > 0
        assert identity.get("isPrivate") is True and identity.get("currentVersionNumber") == 1
        observed = {"id": identity["id"], "slug": DATASET_SLUG, "private": True, "version": 1}
        if (D / "dataset-identity.json").exists():
            assert read(D / "dataset-identity.json") == observed
        else:
            w.new_json(D / "dataset-identity.json", observed)
        state = {"status": "ORIGINAL_CREATED_PRIVATE_IDENTITY_NOT_YET_STATUS_VISIBLE"}"""
    )
    program = replace(program, status_call, adapted)
    program = replace(
        program,
        "    # Compiler bounds its name search",
        '    if (D / "dataset-identity.json").exists():\n        assert read(D / "dataset-identity.json") == {k: result[k] for k in ("id", "slug", "private", "version")}\n    # Compiler bounds its name search',
    )
    # Source of the timer and reported budget must use the same reviewed cap.
    cap, public = plan["qualification_cap_seconds"], plan["public_watchdog_seconds"]
    policy = replace(
        policy,
        'PLAN["qualification_cap_seconds"] != 600',
        'PLAN["qualification_cap_seconds"] != ' + str(cap),
    )
    policy = replace(
        policy,
        'PLAN["public_watchdog_seconds"] != 300',
        'PLAN["public_watchdog_seconds"] != ' + str(public),
    )
    policy = replace(
        policy, "PUBLIC_WATCHDOG_SECONDS = 300", "PUBLIC_WATCHDOG_SECONDS = " + str(public)
    )
    policy = replace(policy, "threading.Timer(300,", "threading.Timer(" + str(public) + ",")
    policy = replace(
        policy,
        "public_example_watchdog_seconds=300",
        "public_example_watchdog_seconds=" + str(public),
    )
    result = {
        n + ".py": parent["scripts/kaggle/" + n + ".py"]
        for n in ("workflow", "components", "gpu_capacity", "output_csv", "publication")
    }
    result["policy.py"], result["program.py"] = policy, program
    for name, text in result.items():
        compile(text, name, "exec")
    return result


def runner_source(parent):
    text = parent[P + "run_once.py"]
    text = replace(text, "actual-pointwise-compile.json", "actual-teacher_rank-compile.json")
    text = replace(text, "rsna-pointwise-serving-", "rsna-teacher-rank-serving-")
    tree = ast.parse(text)
    final = tree.body[-1]
    assert isinstance(final, ast.If) and ast.unparse(final.test) == "__name__ == '__main__'"
    return "\n".join(text.splitlines()[: final.lineno - 1]) + "\n"


def owner_source(parent):
    text = parent[P + "follow_once.py"]
    # Both owners can observe outcomes; only this route's phase mutates under the same claim.
    before = '    source_zero = R / "artifacts/source-zero-final-serving-20261020-v2"'
    start = text.index(before)
    end = text.index("    config = json.loads(args.config.read_text())", start)
    text = (
        text[:start]
        + """    pointwise = R / "initiative/research/pointwise-final-deployment-20261020-v1"
    assert (pointwise / "actual-qualification-accepted.json").is_file()
    assert (pointwise / "actual-competition-accepted.json").is_file()
    assert (pointwise / "actual-entry.json").is_file()
"""
        + text[end:]
    )
    text = replace(
        text,
        '"actual_pointwise_compiled_and_absent_route_reconciled_read_only"',
        '"actual_teacher_rank_compiled_and_absent_route_reconciled_read_only"',
    )
    text = replace(
        text, '"pointwise_dataset_create_returned"', '"teacher_rank_dataset_create_returned"'
    )
    final = ast.parse(text).body[-1]
    assert isinstance(final, ast.If)
    return "\n".join(text.splitlines()[: final.lineno - 1]) + "\n"
