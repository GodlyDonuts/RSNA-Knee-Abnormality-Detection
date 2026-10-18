"""Record one prospective scientific decision from actual immutable receipts.

No cloud access, model fitting, input mutation or automatic diagnostic retry.
Run after both original finite owners have terminal success, never on progress.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rsna.competitive.paired_probe_decision import decide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement", type=Path, required=True)
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Byte identities bind the original owner files; no JSON reserialization is
    # substituted for their hashes. Exclusive output prevents a silent revision.
    measured = args.measurement.read_bytes()
    recovered = args.recovery.read_bytes()
    m, r = json.loads(measured), json.loads(recovered)
    if (
        m["status"] != "new_trained_encoder_noise_measurement_actual_complete"
        or r["status"] != "new_paired_logit_encoder_engineering_trial_actual_complete"
    ):
        raise ValueError("Original owner terminal successes required; progress is not acceptance")
    decision = decide(m["actual"], r["actual"])
    decision["original_measurement_owner_bytes_sha256"] = hashlib.sha256(measured).hexdigest()
    decision["original_recovery_owner_bytes_sha256"] = hashlib.sha256(recovered).hexdigest()
    decision["decision_module_sha256"] = hashlib.sha256(
        (
            Path(__file__).resolve().parents[2] / "rsna/competitive/paired_probe_decision.py"
        ).read_bytes()
    ).hexdigest()
    with args.output.open("x") as stream:
        json.dump(decision, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(decision, allow_nan=False))


if __name__ == "__main__":
    main()
