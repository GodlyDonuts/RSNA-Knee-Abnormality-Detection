"""Bind HTTP recovery and retained ancestral intents without altering staged files."""
import json
import runpy
import types
from pathlib import Path

import successor_contract as c


def validate(terminal, intent, raw, terminal_job, control):
    lineage = terminal["continuation_chain"]
    c.require(intent["continuation_chain"] == raw["continuation_chain"] == lineage,
              "Input continuation lineages differ")
    digest = c.chain(lineage)
    ancestors = lineage["raw"]["ancestor_chain"]
    for mapping in (ancestors, lineage):
        for item in mapping.values():
            actual = terminal_job(str(item["job"]), ("FAILED",))
            c.require(actual["exit_code"] == "1:0", "Retained failure changed")
            old = Path(item["root"]) / "intent.json"
            body = old.read_bytes()
            c.require(c.sha(body) == item["intent_sha256"], "Retained ancestor intent changed")
            if "ancestor_chain" in item:
                c.require(json.loads(body)["continuation_chain"] == item["ancestor_chain"],
                          "Claimed ancestors differ from the actual prior intent")
    raw_final = json.loads((Path(c.ROOTS["raw"]) / "actual.json").read_text())
    c.require(raw_final["continuation_chain"] == lineage
              and raw_final["CPU_job"] == terminal["successor_raw_job"],
              "Successful raw final lineage differs")
    bound = c.proofs(terminal, intent, raw, raw_final)
    c.require(terminal["scratch_cap_bytes"] == intent["scratch_cap_bytes"] ==
              raw["scratch_cap_bytes"] == raw_final["scratch_cap_bytes"] == 6 * 2**30,
              "Retained scratch scope changed")
    return {**control, **bound, "continuation_chain_sha256": digest}


def install(remote, payload):
    original_namespace = remote["namespace"]

    def namespace(value, node):
        ns = original_namespace(dict(value, original_remote=c.derived_remote(value["original_remote"])), node)
        if node != "evc33":
            return ns
        original_run_path = runpy.run_path

        def run_path(path, run_name=None):
            if Path(path) != Path("/tmp/rsna-predictive-full-fitting-20261020-v2/code/admit_cpu.py"):
                return original_run_path(path, run_name=run_name)
            derived = c.derived_admission(Path(path).read_text())
            control = {
                "operational_control_git": payload["placement_source_git"],
                "prior_operational_source_git": c.PRIOR,
                "prior_operational_sources_sha256": c.PRIOR_SHA,
                "original_admission_sha256": c.ADMIT_SHA,
                "derived_admission_sha256": c.sha(derived.encode()),
                "successor_contract_sha256": payload["successor_code_sha256"]["contract.py"],
                "remote_adapter_sha256": payload["successor_code_sha256"]["remote_adapter.py"],
                "original_scientific_preparation_reused": True,
                "scientific_source_git": c.SCIENCE,
                "retained_total_scratch_bytes": 6 * 2**30,
            }
            scope = {"__name__": run_name or "http_successor_admission", "__file__": str(path),
                     "validate_successor_chain": lambda t, i, r, j: validate(t, i, r, j, control)}
            exec(compile(derived, str(path) + ":http_control_adapter", "exec"), scope)
            return scope

        ns["runpy"] = types.SimpleNamespace(run_path=run_path)
        return ns

    remote["namespace"] = namespace
