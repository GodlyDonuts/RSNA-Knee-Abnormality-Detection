"""Resume exact original primary phases after its one returned Dataset creation."""

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PLACEMENT = ROOT / "initiative/research/predictive-evc27-deadline-placement-20261020-v1"
ORIGINAL = ROOT / "artifacts" / PLACEMENT.name
OLD_JOURNAL = ORIGINAL.with_name(ORIGINAL.name + "-deployment")
JOURNAL = ROOT / "artifacts" / HERE.name
PLACEMENT_PIN = "2a224ae05fc7c2fc78d4d508b6407344ac523a1c"
DATASET = "godlydonuts/rsna-predictive-finals-5179059eb7cf"
DATASET_ID = 12756510


def read(path):
    return json.loads(path.read_bytes())


def source(pin, path):
    return subprocess.check_output(["git", "show", pin + ":" + str(path.relative_to(ROOT))], cwd=ROOT)


def module(body, name, path):
    value = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, None))
    value.__file__ = str(path)
    exec(compile(body, str(path), "exec"), value.__dict__)
    return value


def remaining_owner(body):
    start = body.index("    def run(self):\n")
    end = body.index('        self.action("dataset-reconcile")\n', start)
    prefix = body[start:end]
    assert prefix.count('self.action("stage")') == 1
    assert prefix.count('self.action("prepare")') == 1
    assert prefix.count('self.action("dataset")') == 1
    return body[:start] + "    def run(self):\n        self.admit_existing_dataset()\n" + body[end:]


def validate_original(journal, actual, command):
    owner = read(journal / "owner-intent.json")
    assert owner["pid"] == 32378 and owner["placement_source_git"] == PLACEMENT_PIN
    assert (journal / "stopped.json").is_file()
    assert command(32378).strip() == "", "Original PID must have ended"
    assert read(journal / "stage-qualification-result.json")["status"] == "predictive_final_deployment_controls_staged"
    prepared = read(journal / "prepare-qualification-result.json")
    assert prepared["status"] == "actual_three_predictive_final_exports_prepared_CPU_only"
    assert prepared["dataset_slug"] == DATASET
    created = read(journal / "dataset-qualification-result.json")
    assert created == dict(status="predictive_dataset_create_returned", new_Dataset_calls=1, new_GPU_or_entries=0)
    assert actual["dataset_slug"] == DATASET and actual["bundle_identity"] == prepared["bundle_identity"]
    assert actual["claim_present"] is False and actual["mutations"] == 0
    metadata = actual["responses"]["GetDataset"]
    assert metadata["HTTP"] == 200 and metadata["id"] == DATASET_ID
    assert metadata["isPrivate"] is True and metadata["currentVersionNumber"] == 1
    assert actual["responses"]["GetDatasetStatus"] == dict(HTTP=200, result={"status": "READY"})
    assert len(actual["ledger"]) == 1
    row = actual["ledger"][0]["value"]
    assert row["state"] == "predictive_dataset_create_returned" and row["dataset_slug"] == DATASET
    assert actual["dataset-response.json"]["final_create_rpc_calls"] == 1
    for name in ("dataset-reconcile", "compile", "push", "submit"):
        assert not list(journal.glob(name + "-*-intent.json")), "Prior phase requires separate reconciliation"
    assert not any(x.startswith("compiled-") for x in actual["root_files"])
    return prepared


def build():
    contract = module(source(PLACEMENT_PIN, PLACEMENT / "contract.py"), "continue_placement_contract", PLACEMENT / "contract.py")
    import sys
    old = sys.modules.get("contract")
    sys.modules["contract"] = contract
    try:
        fitting = module(source(PLACEMENT_PIN, PLACEMENT / "owner.py"), "continue_placement_owner", PLACEMENT / "owner.py")
    finally:
        if old is None:
            sys.modules.pop("contract", None)
        else:
            sys.modules["contract"] = old
    with fitting.imports(contract=contract, owner=fitting):
        dep = module(source(PLACEMENT_PIN, PLACEMENT / "deployment.py"), "continue_placement_deployment", PLACEMENT / "deployment.py")
    _, lineage = fitting.previous()
    path = ROOT / dep.CANONICAL_DIR / "owner.py"
    body = dep.final_reader(lineage.derived_deployment_owner(source(dep.CANONICAL, path).decode()))
    original = module(remaining_owner(body), "continue_primary_owner", path)
    reader = type("Reader", (), {"full_finals": staticmethod(original.full_finals)})
    original.full_finals = lambda: dep.finals(reader, lineage)
    placement_body = fitting.git_file(lineage.PARENT, lineage.OLD + "/placement.py").decode()
    return fitting, dep, original, reader, lineage, placement_body


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    os.umask(0o077)
    assert source(args.source_git, Path(__file__)) == Path(__file__).read_bytes()
    diagnostic = read(JOURNAL / "original-readonly.json")
    assert diagnostic["returncode"] == 0
    actual = json.loads(diagnostic["stdout"])
    def command(pid):
        return subprocess.run(["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True).stdout
    prepared = validate_original(OLD_JOURNAL, actual, command)
    fitting, dep, original, reader, lineage, placement_body = build()
    decision = read(PLACEMENT / "deployment-decision.accepted.json")
    original.validate_decision(decision)
    lineage.deployment_funding(decision)
    assert decision["portable_placement_source_git"] == PLACEMENT_PIN
    assert original.full_finals() is not None
    original.once(JOURNAL / "continuation-intent.json", dict(
        pid=os.getpid(), source_git=args.source_git, original_owner_pid=32378,
        original_placement_source_git=PLACEMENT_PIN, canonical_source_git=dep.CANONICAL,
        Dataset_id=DATASET_ID, Dataset_slug=DATASET, new_Dataset_calls=0,
        original_phases_preserved=True, automatic_mutation_retry=False))

    class Continuation(original.Owner):
        def _observe(self, dataset_slug=None):
            values = dep.finals(reader, lineage)
            assert values is not None
            jobs = read(PLACEMENT / "actual-known-dispatches-all27.json")["jobs"]
            assert all(jobs[a]["GPU_job"] == values["finals"][a]["GPU_job"] for a in original.ARMS)
            self.observer = dep.observer_source(
                fitting.git_file(dep.CANONICAL, dep.CANONICAL_DIR + "/observe.py.txt").decode(),
                values, jobs, placement_body)
            return super()._observe(dataset_slug)

        def config(self, observed, mode):
            return dep.rebind_config(super().config(observed, mode), observed["paid_original_GPU_logs"])

        def admit_existing_dataset(self):
            observed = self.observe(DATASET)
            assert observed["dataset_ready"] and observed["dataset_id"] == DATASET_ID
            assert not observed["deployment_root_absent"] and not observed["canonical_claim_present"]
            original.once(JOURNAL / "existing-dataset-admitted.json", observed)
            assert prepared["bundle_identity"] == actual["bundle_identity"]

    try:
        Continuation(dep.CANONICAL, decision, args.socket, JOURNAL).run()
    except BaseException as exc:
        original.once(JOURNAL / "continuation-stopped.json", dict(error_type=type(exc).__name__, automatic_retry=False))
        raise


if __name__ == "__main__":
    main()
