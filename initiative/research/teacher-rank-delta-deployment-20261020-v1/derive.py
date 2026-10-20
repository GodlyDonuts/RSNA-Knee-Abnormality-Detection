"""Operational single-run delta over the immutable prepared ranking account route."""

import ast
import hashlib
import json
import subprocess

RANK_SOURCE = "16a35eca2159e01dea22511d9f545a339cfe896b"
RANK_DIR = "initiative/research/teacher-rank-final-deployment-20261020-v1/"
NAME = "teacher-rank-delta-deployment-20261020-v1"
SERVING = "initiative/research/teacher-rank-delta-serving-20261020-v1/"
INHERITED_PATH = (
    "initiative/research/pointwise-final-deployment-20261020-v1/"
    "actual-qualification-accepted.json"
)
INHERITED_SHA = "13d7e576e15a9036ebf463ea3a74b5ff39c6c836a1f3d2bef9f49274a8c7463a"
MODEL = "d478a502e6fd33be2ad9d781acd91d3d2b6b252629262031fba8282a91cf5bca"
SCOPE = "inherited32_same_code_plus_actual_public3_new_weights"


def exact(source, before, after):
    assert source.count(before) == 1, "One exact frozen operational site required"
    return source.replace(before, after)


def rank_parent(repo):
    def read(path):
        return subprocess.check_output(["git", "show", RANK_SOURCE + ":" + path], cwd=repo)

    body = read(RANK_DIR + "derive.py")
    ns = {"__name__": "frozen_rank_derivation"}
    exec(compile(body, "frozen_rank_derivation", "exec"), ns)
    manifest = json.loads(read(RANK_DIR + "frozen-parent.json"))
    parent = ns["parent_files"](repo, manifest)
    return ns, parent, {
        "rank_source_git": RANK_SOURCE,
        "rank_derive_sha256": hashlib.sha256(body).hexdigest(),
        "rank_parent_manifest_sha256": hashlib.sha256(
            read(RANK_DIR + "frozen-parent.json")
        ).hexdigest(),
        "pointwise_source_git": ns["PARENT"],
        "pointwise_file_sha256": manifest["files"],
    }


def accepted(config, decision_text):
    plan, decision = config["runtime_plan"], json.loads(decision_text)
    assert plan["status"] == "accepted_teacher_rank_delta_runtime_budget"
    assert plan["engineering_decision_sha256"] == hashlib.sha256(
        decision_text.encode()
    ).hexdigest()
    assert decision["status"] == "accepted_teacher_rank_delta_runtime_and_storage_plan"
    assert decision["model_sha256"] == MODEL
    for k, value in {
        "public_watchdog_seconds": 180,
        "public_VM_setup_reserve_seconds": 80,
        "predictive_reserved_VM_seconds": 3360,
        "required_free_VM_seconds": 3620,
    }.items():
        assert type(plan[k]) is int and plan[k] == decision[k] == value
    assert "qualification_cap_seconds" not in plan
    assert plan["inherited32_receipt_sha256"] == INHERITED_SHA
    assert plan["qualification_scope"] == SCOPE
    assert decision["evidence_sha256"][INHERITED_PATH] == INHERITED_SHA
    assert decision["qualification_scope"]["new_weight_32_case_pass_claim"] is False
    operations = decision["operational_contract"]
    assert operations["old_prepared_ranking_route_remains_unexecuted"] is True
    assert decision["qualification_scope"][
        "single_public_execution_and_same_immutable_version_submission"
    ] is True
    assert all(operations[k] == 1 for k in (
        "maximum_new_Datasets", "maximum_new_GPU_sessions", "maximum_new_entries"
    ))
    assert operations["no_uncertain_retry"] is True
    assert operations["separate_qualification_status_invented"] is False
    funding = config["funding"]
    assert funding["incremental_node_bytes"] == 640 * 2**20
    assert funding["incremental_shared_bytes"] == 2**20
    assert funding["incremental_shared_files"] == 0
    assert funding["shared_other_liabilities_bytes"] == 122 * 2**20
    assert funding["shared_future_files"] == 376
    return plan


def sources(repo, config, decision_text, policy_text, serving_files, inherited_text):
    plan = accepted(config, decision_text)
    assert hashlib.sha256(inherited_text.encode()).hexdigest() == INHERITED_SHA
    rank, parent, provenance = rank_parent(repo)
    # Only an intermediate source generator consumes this unused historical cap.
    # Its entire budget/request/timer path is removed below; no qualification is run.
    rank["accepted"] = lambda *_: {**plan, "qualification_cap_seconds": 600}
    result = rank["sources"](parent, config, decision_text)
    policy = result["policy.py"]
    for name in ("budget", "request", "public_bounded_source"):
        node = next(n for n in ast.parse(policy).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
        lines = policy.splitlines(keepends=True)
        policy = "".join(lines[:node.lineno - 1] + lines[node.end_lineno:])
    policy = exact(policy, "def receipt(r, candidate, mode):",
                   "def _original_receipt(r, candidate, mode):")
    policy += "\nINHERITED_QUALIFICATION_TEXT = " + repr(inherited_text) + "\n" + policy_text
    program = exact(
        result["program.py"],
        'D = Path("/tmp/rsna-teacher-rank-final-deployment-20261020-v1")',
        'D = Path("/tmp/rsna-' + NAME + '")',
    )
    program = exact(
        program, "def inspect_original(v):",
        'def inspect_original(v):\n'
        '    assert not Path("/tmp/rsna-teacher-rank-final-deployment-20261020-v1").exists(), '
        '"Original prepared ranking route must remain absent"',
    )
    program = exact(
        program, "        prepare_bundle(original)",
        '        preflight_policy = load("policy")\n'
        '        preflight_source, preflight_metadata = preflight_policy.adapt_source(\n'
        '            (ORIGINAL_SERVING / "competition.private.py").read_text(), "competition")\n'
        '        preflight = dict(status="actual_original_rank_delta_source_compiled_before_bundle",\n'
        '                         source_sha256=hashlib.sha256(preflight_source.encode()).hexdigest(),\n'
        '                         source_bytes=len(preflight_source.encode()), metadata=preflight_metadata,\n'
        '                         new_Dataset_GPU_entry_calls=0)\n'
        '        assert preflight["source_sha256"] == preflight_metadata["source_sha256"]\n'
        '        with (F / "delta-source-preflight.json").open("x") as handle:\n'
        '            json.dump(preflight, handle, sort_keys=True)\n'
        '        out["delta_preflight"] = preflight\n'
        '        del preflight_source\n'
        '        prepare_bundle(original)',
    )
    program = exact(program, "                texts = {}",
                    "                texts = {}\n                delta_metadata = {}")
    program = exact(
        program,
        "                    texts[mode] = policy.public_bounded_source(path.read_text(), mode)",
        "                    texts[mode], delta_metadata[mode] = policy.adapt_source("
        "path.read_text(), mode)\n"
        '                    assert delta_metadata[mode] == read(F / "delta-source-preflight.json")["metadata"]',
    )
    program = exact(
        program,
        '                    (target / (mode + ".private.py")).write_text(text)',
        '                    (target / (mode + ".private.py")).write_text(text)\n'
        '                    assert delta_metadata[mode]["source_sha256"] == sources[mode]["sha256"]\n'
        '                    w.new_json(target / (mode + "-delta-binding.json"), delta_metadata[mode])',
    )
    program = exact(program, '                    "source": sources,',
                    '                    "source": sources,\n'
                    '                    "delta_metadata": delta_metadata,')
    program = exact(
        program, '            assert source_sha == row["source"][mode]["sha256"]',
        '            assert source_sha == row["source"][mode]["sha256"]\n'
        '            metadata = read(target / (mode + "-delta-binding.json"))\n'
        '            assert metadata == row["delta_metadata"][mode]\n'
        '            assert metadata["source_sha256"] == source_sha',
    )
    program = exact(
        program,
        '                assert row["state"] == (\n'
        '                    "full_parent_epoch_model_verified"\n'
        '                    if mode == "qualification"\n'
        '                    else "full_parent_epoch_qualification_accepted"\n'
        '                )',
        '                assert mode == "competition"\n'
        '                assert row["state"] == "full_parent_epoch_model_verified"',
    )
    program = exact(program, "                    policy.receipt(receipt, candidate, mode)",
                    "                    policy.receipt(receipt, candidate, mode, metadata)")
    program = exact(
        program, '                account = remote.account()',
        '                policy.receipt(public["receipt"], candidate, mode, metadata)\n'
        '                assert public["CSV_verified_in_RAM"]["CSV_sha256"] == public["receipt"]["CSV_sha256"]\n'
        '                account = remote.account()',
    )
    provenance["delta_source_sha256"] = {
        name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in serving_files.items()
    }
    program = exact(
        program, '        out.update(runtime_plan=v["runtime_plan"]',
        '        out["frozen_dependencies"] = ' + repr(provenance) + '\n'
        '        out.update(runtime_plan=v["runtime_plan"]',
    )
    result.update({"policy.py": policy, "program.py": program, **serving_files})
    for name, source in result.items():
        compile(source, name, "exec")
    return result


def runner_source(repo):
    rank, parent, _ = rank_parent(repo)
    text = rank["runner_source"](parent)
    text = exact(text, 'assert mode in {"qualification", "competition"}',
                 'assert mode == "competition"')
    text = exact(text, 'parser.add_argument("--mode", default="qualification")',
                 'parser.add_argument("--mode", choices=["competition"], default="competition")')
    return text


def owner_source(repo):
    rank, parent, _ = rank_parent(repo)
    text = rank["owner_source"](parent)
    text = exact(text, '    config = json.loads(args.config.read_text())',
                 '    assert not (R / "artifacts/teacher-rank-final-deployment-20261020-v1").exists()\n'
                 '    config = json.loads(args.config.read_text())')
    text = exact(text, "maximum_new_GPU_sessions=2", "maximum_new_GPU_sessions=1")
    text = exact(text, 'def action(phase, mode="qualification"):',
                 'def action(phase, mode="competition"):')
    text = exact(text, 'for mode in ("qualification", "competition"):',
                 'for mode in ("competition",):')
    return text
