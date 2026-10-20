"""Reuse immutable C100 operations with explicit, reviewable G100 substitutions."""

import ast
import inspect
import subprocess
from pathlib import Path

PIN = "473a9a94d90890e8a97e7ed46d157dcf417ad173"
ORIGINAL = "initiative/research/predictive-density100-full-20261020-v1/"


def replace_function(text, name, new):
    tree = ast.parse(text)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1
    n = nodes[0]
    lines = text.splitlines(keepends=True)
    return "".join(lines[: n.lineno - 1]) + new.rstrip() + "\n" + "".join(lines[n.end_lineno :])


VALIDATE_DECISION_SOURCE = '''
def validate_decision(decision, pilot, pilot_sha, source_git, code_sha):
    # The generated admission supplies these functions from the pinned source.
    validate_pilot(pilot, decision["pilot_scheduler"])
    require(
        decision["status"] == "accepted_separate_full_G100_after_actual_pilot_review"
        and decision["branch"] == "G100"
        and decision["pilot_acceptance_sha256"] == pilot_sha
        and decision["source_git"] == source_git
        and decision["code_sha256"] == code_sha
        and decision["recipe"] == recipe()
        and decision["matched_peak_LR_diagnostics_reviewed"] is True
        and decision["observed_C_C0_gradient_risk_reviewed"] is True
        and decision["primary_B_C_C0_unchanged"] is True
        and decision["primary_Kaggle_reserve_seconds"] == 3360
        and decision["engineering_predictions_used_for_selection"] is False
        and decision["automatic_recipe_retuning"] is False
        and isinstance(decision["agent_reasoning"], str)
        and len(decision["agent_reasoning"]) >= 80,
        "Separate full-data G100 decision must bind actual matched diagnostics",
    )
'''


def sources(repo, contract):
    repo = Path(repo)

    def original(name):
        return subprocess.check_output(
            ["git", "show", PIN + ":" + ORIGINAL + name], cwd=repo
        ).decode()

    texts = {name: original(name) for name in ("admission.py", "operations.py", "coordinator.py")}
    for name, body in texts.items():
        # These exact originals are never modified. The resulting G100 capsule
        # has separate paths, journal, job names, evidence, funding and epochs.
        texts[name] = (
            body.replace("C100", "G100")
            .replace("density100", "generative")
            .replace("981620", "986011")
        )
    a = texts["admission.py"]
    replacements = {
        'PILOT_SOURCE = "068a14a75e4d2b3bd83dda7783528956f7a2ca0f"': f'PILOT_SOURCE = "{contract.PILOT_SOURCE}"',
        'PILOT_BINDING = "e11c7c7fe5c1fd7cae87df313fb5d42be7128b7906f8d716a486b0857bbdd37a"': f'PILOT_BINDING = "{contract.PILOT_BINDING}"',
        'PILOT = Path("/tmp/rsna-predictive-generative-pilot-20261020-v1")': 'PILOT = Path("/tmp/rsna-predictive-generative-routing-20261020-v1")',
    }
    for old, new in replacements.items():
        assert a.count(old) == 1
        a = a.replace(old, new, 1)
    a = replace_function(a, "validate_pilot", inspect.getsource(contract.validate_pilot))
    a = replace_function(a, "validate_decision", VALIDATE_DECISION_SOURCE)
    # In C100's old budget the extra six-GiB scope belonged to the G100 pilot.
    # G100 now charges its own pilot by actual original bytes, and retains the
    # C100 pilot plus the *entire* separate C100 full-run reservation instead.
    a = a.replace("G100_RESERVED_BYTES", "C100_PILOT_RESERVED_BYTES")
    a = a.replace(
        "conservative_G100_reserved_bytes_no_paid_credit",
        "C100_pilot_reserved_bytes_no_paid_credit",
    )
    old = "CONTRAST_RESERVED_BYTES = 7 * GIB + 4 * MIB\n"
    assert a.count(old) == 1
    a = a.replace(old, old + "C100_FULL_RESERVED_BYTES = 12 * GIB + 16 * MIB\n", 1)
    old = "+ PRIMARY_OTHER_FITS_RESERVED_BYTES + PRIMARY_ADDITIONAL_CONTROLS_RESERVED_BYTES\n"
    assert a.count(old) == 1
    a = a.replace(
        old,
        "+ PRIMARY_OTHER_FITS_RESERVED_BYTES + PRIMARY_ADDITIONAL_CONTROLS_RESERVED_BYTES\n"
        "             + C100_FULL_RESERVED_BYTES\n",
        1,
    )
    old = '"contrast_reserved_bytes_no_paid_credit": CONTRAST_RESERVED_BYTES,\n'
    assert a.count(old) == 1
    a = a.replace(
        old,
        old + '            "C100_full_reserved_bytes_no_paid_credit": C100_FULL_RESERVED_BYTES,\n',
        1,
    )
    texts["admission.py"] = a
    for name, text in texts.items():
        compile(text, "derived_G100_" + name, "exec")
    return texts
