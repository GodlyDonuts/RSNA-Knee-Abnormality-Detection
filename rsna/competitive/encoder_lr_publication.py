"""Keep original and new checkpoint receipt contracts separate in storage accounting."""

ORIGINAL_STATUS = "actual_matched_IID16_parent_epoch_private_full_download_size_SHA_verified"
TREATMENT_STATUS = "actual_encoderLR3e5_epoch_private_full_download_size_SHA_verified"


def sum_checkpoint_receipts(rows, *, collection):
    if collection not in ("original_iid16", "encoder_lr"):
        raise ValueError("Explicit original or treatment checkpoint collection required")
    expected = ORIGINAL_STATUS if collection == "original_iid16" else TREATMENT_STATUS
    seen = set()
    total = 0
    for row in rows:
        role, epoch, size = row.get("role"), row.get("epoch"), row.get("retained_original_bytes")
        proof = row.get("independent_CPU_state_verification", {})
        if (
            row.get("status") != expected
            or role
            not in (("uniform", "influence") if collection == "original_iid16" else ("influence",))
            or type(epoch) is not int
            or not 1 <= epoch <= 3
            or row.get("completed") != 486 * epoch
            or type(size) is not int
            or not 0 < size < 2 * 2**30
            or (role, epoch) in seen
            or proof.get("full_learned_graph_exact") is not True
            or proof.get("finite_contiguous_updates") != 486 * epoch
            or proof.get("actual_AdamW_parameters") != 73136096
        ):
            raise ValueError("Checkpoint bytes require their own exact valid collection receipt")
        seen.add((role, epoch))
        total += size
    if collection == "original_iid16":
        if seen != {(role, epoch) for role in ("uniform", "influence") for epoch in (1, 2, 3)}:
            raise ValueError(
                "All six completed original receipts required for original funding credit"
            )
        cap = 12 * 2**30
    else:
        if seen != {("influence", epoch) for epoch in range(1, len(seen) + 1)}:
            raise ValueError("New treatment publication receipts must be a contiguous prefix")
        cap = 4 * 2**30
    if total >= cap:
        raise ValueError("Retained receipt bytes exceed their separately funded collection cap")
    return total
