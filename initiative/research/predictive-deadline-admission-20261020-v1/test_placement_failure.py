import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "predictive_placement_failure", Path(__file__).with_name("placement_failure.py")
)
policy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy)


def jobs(*arms):
    return {
        arm: {"GPU_job": str(100 + i), "arm": arm, "node": "evc27" if arm == "C0" else "evc33",
              "source_git": policy.SCIENCE, "binding_sha256": "a" * 64}
        for i, arm in enumerate(arms)
    }


@pytest.mark.parametrize("phase", policy.PHASES)
def test_placement_exception_keeps_B_C_and_does_not_invent_C0_failure(phase):
    original = jobs("B", "C")
    result = policy.continuation_after_exception(original, phase=phase, error_type="TimeoutExpired")
    assert result["observe_and_publish_original_jobs"] == original
    assert result["must_continue_B_C_observation_and_epoch_publication"]
    assert result["C0_operation_state"] == "requires_read_only_reconciliation"
    assert not result["C0_GPU_job_known"]
    assert not result["C0_model_or_Slurm_failure_inferred"]
    assert not result["C0_automatic_transfer_retry"]
    assert not result["C0_automatic_dispatch_retry"]
    assert not result["C0_automatic_return_retry"]
    assert not result["automatic_training_restart"]
    assert result["remote_job_cancellations"] == []
    assert not result["authorizes_new_external_mutation"]
    result["observe_and_publish_original_jobs"]["B"]["GPU_job"] = "999"
    assert original["B"]["GPU_job"] == "100"


def test_durable_C0_receipt_is_observed_even_when_local_operation_failed():
    original = jobs("B", "C", "C0")
    result = policy.continuation_after_exception(
        original, phase="dispatch-C0", error_type="ConnectionError",
    )
    assert result["C0_GPU_job_known"]
    assert result["observe_and_publish_original_jobs"]["C0"] == original["C0"]
    assert not result["C0_model_or_Slurm_failure_inferred"]
    assert not result["C0_automatic_dispatch_retry"]


def test_no_transition_can_replace_missing_B_C_dispatch_proof():
    with pytest.raises(ValueError, match="Already-dispatched"):
        policy.continuation_after_exception(jobs("B"), phase="contexts-C0", error_type="ValueError")


def test_wrong_node_or_reused_job_cannot_be_preserved_as_a_distinct_arm():
    original = jobs("B", "C")
    original["C"]["GPU_job"] = original["B"]["GPU_job"]
    with pytest.raises(ValueError, match="receipt differs"):
        policy.continuation_after_exception(original, phase="contexts-C0", error_type="ValueError")
    original = jobs("B", "C", "C0")
    original["C0"]["node"] = "evc33"
    with pytest.raises(ValueError, match="receipt differs"):
        policy.continuation_after_exception(original, phase="dispatch-C0", error_type="ValueError")


def test_no_arbitrary_error_body_or_other_phase_is_accepted():
    with pytest.raises(ValueError, match="sanitized"):
        policy.continuation_after_exception(
            jobs("B", "C"), phase="contexts-C0", error_type="/private/pixel/path failed",
        )
    with pytest.raises(ValueError, match="Exact placement phase"):
        policy.continuation_after_exception(
            jobs("B", "C"), phase="publish-B-001", error_type="ValueError",
        )
