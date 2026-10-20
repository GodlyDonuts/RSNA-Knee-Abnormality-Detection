"""In-memory operational admission adapter; never replaces staged source files."""
import json
import runpy
import types
from pathlib import Path

import successor_contract as c


def install(remote, payload):
    original_namespace = remote["namespace"]

    def namespace(v, node):
        changed = dict(v, original_remote=c.derived_remote(v["original_remote"]))
        ns = original_namespace(changed, node)
        if node != "evc33":
            return ns
        original_run_path = runpy.run_path

        def run_path(path, run_name=None):
            if Path(path) != Path("/tmp/rsna-predictive-full-fitting-20261020-v2/code/admit_cpu.py"):
                return original_run_path(path, run_name=run_name)
            source = Path(path).read_text()
            derived = c.derived_admission(source)
            control = {
                "operational_control_git": payload["placement_source_git"],
                "original_admission_sha256": c.ADMIT_SHA,
                "derived_admission_sha256": c.sha(derived.encode()),
                "successor_contract_sha256": payload["successor_code_sha256"]["contract.py"],
                "remote_adapter_sha256": payload["successor_code_sha256"]["remote_adapter.py"],
                "original_scientific_preparation_reused": True,
                "scientific_source_git": c.SCIENCE,
                "retained_total_scratch_bytes": 6 * 2**30,
            }

            def validate(terminal, intent, raw, terminal_job):
                lineage = terminal["continuation_chain"]
                c.require(intent["continuation_chain"] == raw["continuation_chain"] == lineage,
                          "Raw/context intent and final continuation lineage differ")
                digest = c.chain(lineage)
                for role, job in c.PREDECESSORS.items():
                    actual = terminal_job(job, ("FAILED",))
                    c.require(actual["exit_code"] == "1:0", "Predecessor failure changed")
                    old_intent = Path(c.OLD_ROOTS[role]) / "intent.json"
                    c.require(c.sha(old_intent.read_bytes()) == c.INTENTS[role],
                              "Retained predecessor intent changed")
                for key in ("capacity_decision_sha256", "mixed_prefix_admission_sha256"):
                    c.require(terminal[key] == intent[key] == raw[key]
                              and len(terminal[key]) == 64, "Successor admission binding differs")
                raw_final = json.loads((Path(c.ROOTS["raw"]) / "actual.json").read_text())
                c.require(raw_final["continuation_chain"] == lineage
                          and raw_final["CPU_job"] == terminal["successor_raw_job"],
                          "Actual successful raw successor lineage differs")
                return {**control, "continuation_chain_sha256": digest,
                        "capacity_decision_sha256": terminal["capacity_decision_sha256"],
                        "mixed_prefix_admission_sha256": terminal["mixed_prefix_admission_sha256"]}

            scope = {"__name__": run_name or "successor_admission", "__file__": str(path),
                     "validate_successor_chain": validate}
            exec(compile(derived, str(path) + ":explicit_control_adapter", "exec"), scope)
            return scope

        ns["runpy"] = types.SimpleNamespace(run_path=run_path)
        return ns

    remote["namespace"] = namespace
