"""One exact public ranking run, with inherited code and actual new-weight checks."""

import hashlib
import importlib.util
import math
from pathlib import Path

CAPS = {"competition": 32400}
PLAN = None
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f"
OFFLINE = "godlydonuts/rsna-paired-full-e3-7e6aa56c53a3"
INHERITED_SHA = "13d7e576e15a9036ebf463ea3a74b5ff39c6c836a1f3d2bef9f49274a8c7463a"
SCOPE = "inherited32_same_code_plus_actual_public3_new_weights"


def budget(quota, epoch, mode, *, latest_training_epoch):
    if epoch != 3 or latest_training_epoch != 3 or mode != "competition":
        raise ValueError("Only the original H100 final ranking competition run is admitted")
    if PLAN is None or PLAN.get("status") != "accepted_teacher_rank_delta_runtime_budget":
        raise ValueError("Exact accepted delta budget required")
    expected = {
        "public_watchdog_seconds": 180,
        "public_VM_setup_reserve_seconds": 80,
        "predictive_reserved_VM_seconds": 3360,
        "required_free_VM_seconds": 3620,
    }
    if any(type(PLAN.get(k)) is not int or PLAN[k] != v for k, v in expected.items()):
        raise ValueError("The public check, startup and predictive reserve must all be funded")
    if PLAN.get("inherited32_receipt_sha256") != INHERITED_SHA or PLAN.get(
        "qualification_scope"
    ) != SCOPE:
        raise ValueError("Inherited code evidence cannot become new-weight qualification")
    values = [
        float(str(quota[k]).removesuffix("s"))
        for k in ("totalTimeAllowed", "timeUsed", "timeReserved")
    ]
    if any(not math.isfinite(x) or x < 0 for x in values):
        raise ValueError("Finite authoritative quota required")
    free = values[0] - values[1] - values[2]
    return dict(
        available=free >= 3620,
        unreserved_VM_seconds=free,
        required_VM_seconds=3620,
        predictive_reserved_VM_seconds=3360,
        public_VM_setup_reserve_seconds=80,
        public_example_watchdog_seconds=180,
        session_timeout_seconds=32400,
        hidden_session_timeout_seconds=32400,
        minimumTimeAllowed_used_as_ceiling=False,
        paid_credits_allowed=False,
    )


def request(epoch, mode, source, training, *, dataset):
    if (
        epoch != 3
        or mode != "competition"
        or training.get("epoch") != 3
        or training.get("status") != "COMPLETE"
        or training.get("kernel_id") is not None
        or training.get("training_Slurm_job") != "975297"
        or training.get("role") != "teacher_rank"
    ):
        raise ValueError("Actual H100 final ranking identity required")
    if (
        dataset.get("status") != "all_payload_bytes_verified"
        or dataset.get("private") is not True
        or dataset.get("version") != 1
        or dataset.get("downloaded_all_bytes") is not True
    ):
        raise ValueError("Private Dataset must have every payload byte verified")
    sha = hashlib.sha256(source.encode()).hexdigest()
    slug = "godlydonuts/rsna-rank-delta-e3-offline-" + sha[:12]
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
        kernelDataSources=[OFFLINE],
        competitionDataSources=["rsna-knee-abnormality-detection"],
        sessionTimeoutSeconds=32400,
        dockerImage=DOCKER,
        dockerImagePinningType="original",
    )


def delta_adapter():
    path = Path(__file__).with_name("rank_delta_adapter.py")
    spec = importlib.util.spec_from_file_location("frozen_rank_delta_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def adapt_source(source, mode):
    if mode != "competition":
        raise ValueError("No separate qualification run")
    return delta_adapter().adapt(
        source, inherited_qualification_text=INHERITED_QUALIFICATION_TEXT  # noqa: F821
    )


def receipt(r, candidate, mode, metadata):
    if mode != "competition":
        raise ValueError("Only the actual same-version public run can qualify")
    _original_receipt(r, candidate, mode)  # noqa: F821
    return delta_adapter().validate_receipt(r, metadata)
