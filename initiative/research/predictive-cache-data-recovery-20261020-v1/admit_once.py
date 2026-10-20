"""Run the frozen missing-only reconciliation once; never dispatch cache jobs."""

import argparse
import hashlib
import importlib.util
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import owner

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def original_arguments():
    values = json.loads((ROOT / "artifacts/predictive-cache-http-recovery-20261020-v1/launcher.json").read_text())
    command = values["command"][2:]
    return dict(zip(command[::2], command[1::2]))


def freeze(pin, mixed=None):
    old = original_arguments()

    def body(key):
        return Path(old["--" + key]).read_bytes()

    def digest(value):
        return hashlib.sha256(value).hexdigest()

    correction = (HERE / "decision.accepted.json").read_bytes()
    regression = (ROOT / "initiative/research/predictive-source-decode-failure-20261020-v1/actual-authoritative-overlay-reproduction.json").read_bytes()
    audit = ROOT / "initiative/audits/source-pixel-representation-v1"
    return owner.freeze(
        pin, body("regression-receipt"), old["--regression-sha256"],
        body("prior-attempt-receipt"), old["--prior-attempt-sha256"],
        body("prefix-admission-receipt"), old["--prefix-admission-sha256"],
        capacity_body=body("capacity-decision"), capacity_sha256=old["--capacity-decision-sha256"],
        http_body=body("http-decision"), http_sha256=old["--http-decision-sha256"],
        correction_body=correction, correction_sha256=digest(correction),
        plan_body=(audit / "source_correction_plan.json").read_bytes(),
        receipt_body=(audit / "hf-source-correction-receipt.json").read_bytes(),
        correction_regression_body=regression, correction_regression_sha256=digest(regression),
        corrected_sources=json.loads((HERE / "actual-authoritative-source-resident.json").read_text())["corrected_sources"],
        mixed_body=mixed, mixed_sha256=None if mixed is None else digest(mixed),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    args = parser.parse_args()
    bundle = freeze(args.source_git)
    program, payload = bundle["programs"]["admit"], bundle["payloads"]["raw"]
    assert "token" not in payload
    path = ROOT / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    spec = importlib.util.spec_from_file_location("data_admission_transport", path)
    transport = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transport)
    journal = ROOT / "artifacts/predictive-cache-data-recovery-20261020-v1-admission"
    journal.mkdir(mode=0o700, exist_ok=True)
    with (journal / "intent.json").open("x") as f:
        json.dump(dict(source_git=args.source_git, started_at=datetime.now(timezone.utc).isoformat(),
                       program_sha256=hashlib.sha256(program.encode()).hexdigest(),
                       no_source_mutations=True), f)
    command = ["srun", "--account=skattel", "--partition=normal", "--qos=normal",
               "--nodelist=evc33", "--nodes=1", "--ntasks=1", "--cpus-per-task=1",
               "--gres=none", "--mem=12G", "--time=00:10:00", "--immediate=60",
               "env", "CUDA_VISIBLE_DEVICES=", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
               "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python", "-c", transport.COMMAND]
    run = subprocess.run(
        ["ssh", "-S", "/tmp/rsna-research-20261001-stable.sock", "-o", "BatchMode=yes",
         "newton", shlex.join(command)], input=transport.envelope(program, payload),
        text=True, capture_output=True, timeout=720)
    (journal / "stdout").write_text(run.stdout)
    (journal / "stderr").write_text(run.stderr)
    assert run.returncode == 0, "Read original output before any further action"
    value = json.loads(run.stdout.strip().splitlines()[-1])
    assert value["status"] == "actual_mixed_HTTP_prefix_and_authoritative_overlay_read_only_admitted"
    with (HERE / "actual-mixed-prefix-admission.json").open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
