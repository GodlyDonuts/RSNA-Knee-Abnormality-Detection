"""Narrow account adaptation; original compiled full94 inference stays unchanged."""

import ast
import hashlib

CORE_SHA = "f9430f3c1a98a2a1d383a2e9896d037ac67c60534d30a278a8a88e2dcd5dbb6e"
POLICY_SHA = "5e479d5fc51e6dc334a885285044847071d4eca91f23be1ffb3aa57860390e07"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"


def exact(source, before, after):
    if source.count(before) != 1:
        raise ValueError("One exact original control site required")
    return source.replace(before, after)


def function(source, name, replacement):
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError("One original function required")
    n, lines = nodes[0], source.splitlines(keepends=True)
    return (
        "".join(lines[: n.lineno - 1])
        + replacement.rstrip()
        + "\n"
        + "".join(lines[n.end_lineno :])
    )


def policy(original, helpers):
    if hashlib.sha256(original.encode()).hexdigest() != POLICY_SHA:
        raise ValueError("Exact original serving policy required")
    s = exact(original, "EPOCHS = (1, 3)", "EPOCHS = (3,)\nPLAN = None")
    s = exact(s, '"cross_lineage_equal_rank"', '"pointwise"')
    s = s.replace("actual_complete_target_twoT4_epoch_", "actual_teacher_pair_final_epoch_")
    s = exact(
        s,
        "sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f2f6b083ad2b2f",
        "sha256:" + DOCKER.split("sha256:")[1],
    )
    s = exact(
        s,
        "count=len(pd.read_csv(str(root)+'/test.csv',usecols=['StudyInstanceUID']))\n    if count==3:",
        "ids=pd.read_csv(str(root)+'/test.csv',usecols=['StudyInstanceUID'],dtype={'StudyInstanceUID':str}).StudyInstanceUID.tolist()\n    roster=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()\n    if roster=='e743f6d8dca6eabedec2f05a98cf86ed820ef1fe8a0eb06338af5b377dc8a59b':",
    )
    s = exact(s, "threading.Timer(600,", "threading.Timer(300,")
    s = exact(s, "PUBLIC_WATCHDOG_SECONDS = 600", "PUBLIC_WATCHDOG_SECONDS = 300")
    s = exact(
        s,
        "    import pandas as pd\n    root=find_test_root()",
        "    import pandas as pd\n    import hashlib, json\n    root=find_test_root()",
    )
    s = function(s, "budget", helpers["budget"])
    s = function(s, "request", helpers["request"])
    compile(s, "pointwise_original_full94_policy", "exec")
    return s


def program(original, helper_source):
    if hashlib.sha256(original.encode()).hexdigest() != CORE_SHA:
        raise ValueError("Exact original account program required")
    s = exact(
        original,
        'D = Path("/tmp/rsna-S6-Kaggle-production-route-20261019-v1")',
        'D = Path("/tmp/rsna-pointwise-final-deployment-20261020-v1")',
    )
    s = exact(s, 'F = D / "serving-account-control-v1"', 'F = D / "account-controls"')
    s = function(s, "bounds", "def bounds(v):\n    return pointwise_bounds(v)")
    s = exact(s, "def execute(v):", helper_source + "\n\ndef execute(v):")
    s = exact(
        s,
        'phase in ("stage", "prepare", "push", "status", "submit", "outcome")',
        'phase in ("inspect", "stage", "dataset", "dataset-reconcile", "prepare", "push", "status", "submit", "outcome")',
    )
    s = exact(s, "and epoch in (1, 3)", "and epoch == 3")
    s = exact(
        s,
        '    bounds()\n    if phase == "stage":',
        """    bounds(v)
    original = original_pointwise(v)
    if phase in ("inspect", "stage"):
        inspected = inspect_original(v)
        if phase == "inspect" or inspected["status"].startswith("waiting_"):
            return inspected
    if phase == "stage":""",
    )
    s = exact(
        s, "        F.mkdir(mode=0o700)", "        D.mkdir(mode=0o700)\n        F.mkdir(mode=0o700)"
    )
    s = exact(
        s,
        '        (F / "staged.json").write_text(json.dumps(out) + "\\n")',
        """        out.update(runtime_plan=v["runtime_plan"], funding=v["funding"],
                   original_compiled=v["original_compiled"], ledger_before_bytes=LEDGER.stat().st_size)
        (F / "staged.json").write_text(json.dumps(out) + "\\n")
        prepare_bundle(original)""",
    )
    s = exact(
        s,
        '    w, policy = load("workflow"), load("policy")',
        """    assert stage["runtime_plan"] == v["runtime_plan"] and stage["funding"] == v["funding"]
    w, policy = load("workflow"), load("policy")
    policy.PLAN = stage["runtime_plan"]""",
    )
    start = s.index('    E = D / ("epoch-" + str(epoch))')
    end = s.index("    candidate = {", start)
    s = (
        s[:start]
        + """    S = ORIGINAL_SERVING
    training, pub, compiled = original
"""
        + s[end:]
    )
    s = exact(s, '"role": "cross_lineage_equal_rank"', '"role": "pointwise"')
    s = exact(s, 'key = "licensed-parent-full-epoch-serving:" + model_sha', "key = MODEL_KEY")
    s = exact(s, 'target = S / "account-phase-v1"', 'target = D / "account-phase-v1"')
    s = exact(
        s,
        '            if phase == "prepare":',
        """            if phase in ("dataset", "dataset-reconcile"):
                retain = phase == "dataset"
                result = dataset_phase(v, db, remote, w)
                retain = False
                return result
            dataset = verified_dataset(remote)
            if phase == "prepare":""",
    )
    s = exact(
        s,
        "request = policy.request(epoch, mode, source, training)",
        "request = policy.request(epoch, mode, source, training, dataset=dataset)",
    )
    start = s.index("                complete = [")
    end = s.index("                budget = policy.budget", start)
    s = (
        s[:start]
        + "                complete = [3]  # Actual immutable H100 final admitted above, no fake Kaggle training record.\n"
        + s[end:]
    )
    s = exact(s, '"rsna-complete-twoT4-S6-e"', '"rsna-pointwise-final-e"')
    s = exact(
        s,
        "    finally:\n        if not retain:",
        '    finally:\n        assert LEDGER.stat().st_size - stage["ledger_before_bytes"] <= 2**20, "Canonical ledger growth cap exceeded"\n        if not retain:',
    )
    s = exact(
        s,
        "print(json.dumps(execute(value)), flush=True)",
        "before=LEDGER.stat().st_size; result=execute(value); after=LEDGER.stat().st_size\n        result['canonical_ledger_bytes']={'before':before,'after':after,'delta':after-before,'total_growth_cap':2**20}\n        print(json.dumps(result), flush=True)",
    )
    compile(s, "pointwise_original_canonical_account", "exec")
    return s
