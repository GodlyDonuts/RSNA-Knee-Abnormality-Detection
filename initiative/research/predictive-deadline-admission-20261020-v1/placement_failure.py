"""Pure operational transition after a C0 placement exception; never retry it."""

import copy
import re

SCIENCE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
PHASES = ("contexts-C0", "dispatch-C0", "return-C0")


def continuation_after_exception(dispatched_jobs, *, phase, error_type):
    """Keep actual submitted jobs alive while an original C0 operation is unresolved.

    The caller supplies only already-verified durable dispatch receipts. This
    function does not discover jobs, reinterpret Slurm state, or create authority
    to repeat a missing/uncertain external operation.
    """
    if phase not in PHASES or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", error_type):
        raise ValueError("Exact placement phase and sanitized exception type required")
    if not {"B", "C"} <= dispatched_jobs.keys() or not dispatched_jobs.keys() <= {"B", "C", "C0"}:
        raise ValueError("Already-dispatched original B/C jobs are required")
    seen = set()
    for arm, job in dispatched_jobs.items():
        identity = job["GPU_job"]
        if (
            not isinstance(identity, str) or not identity.isdecimal()
            or identity in seen or job["arm"] != arm
            or job["node"] != ("evc27" if arm == "C0" else "evc33")
            or job["source_git"] != SCIENCE
        ):
            raise ValueError("Original independent arm/node/source/job receipt differs")
        seen.add(identity)
    return {
        "status": "C0_original_placement_operation_unresolved_preserve_dispatched_jobs",
        "phase": phase,
        "error_type": error_type,
        "C0_operation_state": "requires_read_only_reconciliation",
        "C0_GPU_job_known": "C0" in dispatched_jobs,
        "C0_model_or_Slurm_failure_inferred": False,
        "observe_and_publish_original_jobs": copy.deepcopy(dispatched_jobs),
        "must_continue_B_C_observation_and_epoch_publication": True,
        "C0_automatic_transfer_retry": False,
        "C0_automatic_dispatch_retry": False,
        "C0_automatic_return_retry": False,
        "automatic_training_restart": False,
        "remote_job_cancellations": [],
        "authorizes_new_external_mutation": False,
    }
