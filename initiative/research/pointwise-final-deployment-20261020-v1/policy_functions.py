"""Resource admission for the exact compiled pointwise final; no network on import."""

import hashlib
import math

CAPS = {"qualification": 1800, "competition": 32400}
PLAN = None
OFFLINE = "godlydonuts/rsna-paired-full-e3-7e6aa56c53a3"
PIXELS32 = "godlydonuts/rsna-parent-exact-pixels32-09254e97913d"
DOCKER = ""


def budget(quota, epoch, mode, *, latest_training_epoch):
    if epoch != 3 or latest_training_epoch != 3 or mode not in CAPS:
        raise ValueError("Actual fixed-final pointwise only")
    if PLAN is None or PLAN.get("status") != "accepted_pointwise_final_runtime_budget":
        raise ValueError("Explicit evidence-backed runtime plan required")
    if PLAN["qualification_cap_seconds"] != 600:
        raise ValueError("Bounded actual qualification cap required")
    if PLAN["public_watchdog_seconds"] != 300 or PLAN["predictive_reserved_VM_seconds"] < 3000:
        raise ValueError("Original public watchdog and predictive reserve required")
    if (
        len(PLAN["actual_same_graph_timing_receipt_sha256"]) != 64
        or PLAN.get("same32_case_checks_and_inference_unchanged") is not True
    ):
        raise ValueError("Actual same-graph timing and unchanged checks required")
    values = [
        float(str(quota[k]).removesuffix("s"))
        for k in ("totalTimeAllowed", "timeUsed", "timeReserved")
    ]
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Finite authoritative GPU quota required")
    free = values[0] - values[1] - values[2]
    required = PLAN["predictive_reserved_VM_seconds"] + PLAN["public_watchdog_seconds"]
    if mode == "qualification":
        required += PLAN["qualification_cap_seconds"]
    return dict(
        available=free >= required,
        unreserved_VM_seconds=free,
        required_VM_seconds=required,
        predictive_reserved_VM_seconds=PLAN["predictive_reserved_VM_seconds"],
        session_timeout_seconds=PLAN["qualification_cap_seconds"]
        if mode == "qualification"
        else 32400,
        public_example_watchdog_seconds=300,
        hidden_session_timeout_seconds=32400,
        minimumTimeAllowed_used_as_ceiling=False,
        paid_credits_allowed=False,
    )


def request(epoch, mode, source, training, *, dataset):
    if (
        epoch != 3
        or mode not in CAPS
        or training.get("epoch") != 3
        or training.get("status") != "COMPLETE"
        or training.get("kernel_id") is not None
        or training.get("training_Slurm_job") != "975297"
        or training.get("role") != "pointwise"
    ):
        raise ValueError("Actual H100 fixed-final pointwise binding required")
    if (
        dataset.get("status") != "all_payload_bytes_verified"
        or dataset.get("private") is not True
        or dataset.get("version") != 1
        or dataset.get("downloaded_all_bytes") is not True
    ):
        raise ValueError("Actual independently downloaded private model Dataset required")
    sha = hashlib.sha256(source.encode()).hexdigest()
    slug = (
        "godlydonuts/rsna-pointwise-e3-"
        + ("qual-" if mode == "qualification" else "offline-")
        + sha[:12]
    )
    return dict(
        slug=slug,
        newTitle=slug.split("/")[1].replace("-", " "),
        text=source,
        language="python",
        kernelType="script",
        isPrivate=True,
        enableGpu=True,
        machineShape="NvidiaTeslaT4",
        enableInternet=False,
        datasetDataSources=[dataset["slug"]],
        kernelDataSources=[OFFLINE] + ([PIXELS32] if mode == "qualification" else []),
        competitionDataSources=[]
        if mode == "qualification"
        else ["rsna-knee-abnormality-detection"],
        sessionTimeoutSeconds=PLAN["qualification_cap_seconds"]
        if mode == "qualification"
        else 32400,
        dockerImage=DOCKER,
        dockerImagePinningType="original",
    )
