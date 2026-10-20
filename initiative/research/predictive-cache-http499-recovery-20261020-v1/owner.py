"""Finite once-only HTTP499 successor; no actions occur on import."""

import argparse
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import generation

ROOT = Path(__file__).resolve().parents[3]
CONTROL = Path(__file__).resolve().parent
REL = CONTROL.relative_to(ROOT).as_posix()
PREVIOUS = "e9bdd1249fa7cf75f792b3762c2028bd7beb4337"
PREVIOUS_REL = "initiative/research/predictive-cache-data-recovery-20261020-v1"
ORIGINALS = {"raw": "981958", "context": "981960"}
HTTP_MODULE = "rsna/competitive/predictive/source_http.py"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def git_file(pin, name):
    if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("Exact committed source required")
    return subprocess.check_output(["git", "show", pin + ":" + name], cwd=ROOT).decode()


def module(name, source, path, **values):
    obj = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, None))
    obj.__file__ = str(path)
    obj.__dict__.update(values)
    exec(compile(source, name, "exec"), obj.__dict__)
    return obj


def previous():
    gen = module("frozen_data_generation", git_file(PREVIOUS, PREVIOUS_REL + "/generation.py"),
                 ROOT / PREVIOUS_REL / "generation.py")
    source = git_file(PREVIOUS, PREVIOUS_REL + "/owner.py")
    source = generation.replace(source, "import generation\n", "")
    old = module("frozen_data_owner", source, ROOT / PREVIOUS_REL / "owner.py", generation=gen)
    old_previous = old.previous

    def historical_http():
        http = old_previous()
        # The archived helper is already pinned by its archived accepted decision.
        # A new distinct helper must not force that old helper to equal current HEAD.
        text = git_file(old.PREVIOUS, old.PREVIOUS_REL + "/owner.py")
        text = generation.replace(text, "import generation\n", "")
        text = generation.replace(text,
            '    assert http_source.encode() == (ROOT / HTTP_MODULE).read_bytes(), "Frozen HTTP helper required"\n', "")
        return module("frozen_historical_http_owner", text, ROOT / old.PREVIOUS_REL / "owner.py",
                      generation=http.generation)

    old.previous = historical_http
    return old


def previous_bundle():
    old = previous()
    # Read the original explicit receipt arguments; every body is independently
    # checked again by the archived freeze/contract. No old operation is rerun.
    source = git_file(PREVIOUS, PREVIOUS_REL + "/admit_once.py")
    source = generation.replace(source, "import owner\n", "")
    reader = module("frozen_data_receipt_reader", source, ROOT / PREVIOUS_REL / "admit_once.py", owner=old)
    return reader.freeze(PREVIOUS)


def inventory_source(source):
    """Keep one original inventory loop; select the exact table per owning root."""
    source = generation.replace(source, "len(roots) <= 4", "len(roots) <= 5", 2)
    begin, end = source.index("def raw_inventory("), source.index("\ndef context_inventory(")
    body = source[begin:end]
    body = generation.replace(body, "    for root in roots:\n        intents = {}",
        "    for root in roots:\n        active_records = _effective_records if root == HTTP499_DATA_ROOT else records\n"
        "        intents = {}")
    body = generation.replace(body, "for i, record in enumerate(records):",
                              "for i, record in enumerate(active_records):")
    body = body.replace("records[i]", "active_records[i]")
    body = generation.replace(body, "            recon.source_proof_matches(json.loads(gzip.decompress(body)), record)",
        "            source_proof = json.loads(gzip.decompress(body))\n"
        "            recon.source_proof_matches(source_proof, record)\n"
        "            correction = _source_corrections.get(i) if root == HTTP499_DATA_ROOT else None\n"
        "            require(row.get(\"source_correction\") == correction\n"
        "                    and source_proof.get(\"source_correction\") == correction,\n"
        "                    \"Exact completed source correction provenance required\")")
    source = source[:begin] + body + source[end:]
    return source


def freeze(commit, *, decision_body, decision_sha256, failure_body,
           mixed_body=None, mixed_sha256=None):
    bundle = previous_bundle()
    own = {name: git_file(commit, REL + "/" + name)
           for name in ("owner.py", "generation.py", "suffix.py")}
    for name, text in own.items():
        assert text.encode() == (CONTROL / name).read_bytes(), "Exact frozen new controls required"
    helper = git_file(commit, HTTP_MODULE)
    assert helper.encode() == (ROOT / HTTP_MODULE).read_bytes(), "Exact frozen new HTTP helper required"
    assert sha(decision_body) == decision_sha256
    suffix = inventory_source(bundle["payloads"]["raw"]["suffix_reconcile"]) + "\n" + own["suffix.py"]
    common = {
        "source_git": commit, "suffix_reconcile": suffix,
        "suffix_reconcile_sha256": sha(suffix.encode()),
        "previous_source_http_sha256": bundle["payloads"]["raw"]["source_http_sha256"],
        "source_http_sha256": sha(helper.encode()),
        "http499_recovery_decision_body": decision_body.decode(),
        "http499_recovery_decision_sha256": decision_sha256,
        "http499_failure_receipt_body": failure_body.decode(),
        "mixed_prefix_admission": None if mixed_body is None else mixed_body.decode(),
        "mixed_prefix_admission_sha256": mixed_sha256,
    }
    for payload in bundle["payloads"].values():
        payload.update(common)
    scope = {}
    exec(compile(suffix, "HTTP499_frozen_contract", "exec"), scope)
    scope["http499_contract"](bundle["payloads"]["raw"])
    if mixed_body is not None:
        scope["admission_receipt"](bundle["payloads"]["raw"])
    bundle["programs"] = generation.programs(bundle["programs"])
    raw = bundle["payloads"]["raw"]
    raw["files"][HTTP_MODULE] = helper
    raw["module_sha256"][HTTP_MODULE] = sha(helper.encode())
    bundle["commit"] = commit
    bundle["sha256"].update({REL + "/" + n: sha(s.encode()) for n, s in own.items()})
    bundle["sha256"].update({HTTP_MODULE: sha(helper.encode()), "generated:suffix": sha(suffix.encode())})
    bundle["sha256"].update({"generated:" + role: sha(s.encode()) for role, s in bundle["programs"].items()})
    return bundle


def next_action(jobs, phases):
    old = previous()
    http = old.previous()
    source = git_file(http.PREVIOUS, http.PREVIOUS_REL + "/owner.py")
    start, end = source.index("def next_action("), source.index("\n\ndef command(")
    text = source[start:end].replace('(\"979664\", \"979665\")', '(\"981958\", \"981960\")')
    scope = {}
    exec(compile(text, "HTTP499_once_transition", "exec"), scope)
    return scope["next_action"](jobs, phases)


def command(role, transport):
    return [part.replace("-data-recovery-v1", "-http499-recovery-v1")
            for part in previous().command(role, transport)]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("http499-decision", "failure-receipt", "mixed-prefix-admission"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    extra, remainder = parser.parse_known_args()
    assert sha(extra.failure_receipt.read_bytes()) == extra.failure_receipt_sha256
    capacity = previous().previous().previous()
    old = capacity.base_module()
    source = capacity.git_file(capacity.PREVIOUS, capacity.PREVIOUS_REL + "/owner.py")
    body = source[source.index("def main():"):source.index('\n\nif __name__ == "__main__":')]
    body = body.replace("artifacts/predictive-full-cache-alias-recovery-20261020-v3", "artifacts/" + CONTROL.name)
    exec(compile(body, "distinct_HTTP499_owner_loop", "exec"), old.__dict__)
    old.CONTROL, old.ORIGINALS, old.REMOTE_ROOTS = CONTROL, ORIGINALS, generation.ROOTS
    old.FUNDING, old.command, old.next_action = capacity.FUNDING, command, next_action
    old.freeze = lambda commit, *_args: freeze(
        commit, decision_body=extra.http499_decision.read_bytes(),
        decision_sha256=extra.http499_decision_sha256, failure_body=extra.failure_receipt.read_bytes(),
        mixed_body=extra.mixed_prefix_admission.read_bytes(), mixed_sha256=extra.mixed_prefix_admission_sha256)
    sys.argv = [sys.argv[0], *remainder]
    old.main()


if __name__ == "__main__":
    main()
