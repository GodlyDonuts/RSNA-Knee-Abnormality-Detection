"""Exact, explicit old/new renderer compatibility for retained context caches."""

ORIGINAL_SCIENCE = "b361ee40939d875d3adf82d5e04dbf2d7d62b0a0"
CASES = "rsna/competitive/predictive/cases.py"
ORIGINAL_CASES = "c19e3d45b6573dbd9c631f67e2794493e9e9f4a06f6da49df93f16edb6d93cf8"
CORRECTED_CASES = "5f7c81fd270c072fa16ed486735049bb72070a704304b6ee1adc2c4f6d4f3d09"
RENDERERS = {
    "rsna/competitive/predictive/inputs.py",
    CASES,
    "rsna/competitive/predictive/supports.py",
}
INITIAL = "dbf69ff87261f7a0657c866ef8e41db8b125a62c6245d80017945c634e60158b"
REGRESSION_SHA = "946911a47151af80225cf28921876e88e089c35c86e4c11061ca141bbe2e463e"
PREFIX_SHA = "6c7cb68e72d0b3822ebc6894c38815e79988a256613770a5b1a5c19b617258ad"


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def accepted_renderers(original_code, current_code, regression):
    """Only the admitted cases metadata correction can differ numerically."""
    original = {name: value for name, value in original_code.items() if name.startswith("rsna/")}
    current = {name: value for name, value in current_code.items() if name.startswith("rsna/")}
    require(set(original) == set(current), "Original complete numerical module set differs")
    require(original[CASES] == ORIGINAL_CASES, "Original renderer hash differs")
    require(current[CASES] == CORRECTED_CASES, "Unadmitted corrected renderer hash")
    require(
        all(original[name] == current[name] for name in original if name != CASES),
        "A numerical module other than the admitted metadata correction changed",
    )
    require(
        regression["status"] == "actual_duplicate_acquisition_renderer_regression_accepted"
        and str(regression["original_job"]) == "978318"
        and regression["original_cases_sha256"] == ORIGINAL_CASES
        and regression["corrected_cases_sha256"] == CORRECTED_CASES
        and regression["no_source_downloads"] is True
        and regression["prior_successful_contexts_unchanged"] is True
        and regression["retained_failed_case_corrected"] is True,
        "Actual retained-case and prior-output regression proof required",
    )
    require(RENDERERS <= original.keys(), "Complete original renderer manifest required")
    return [
        {name: original[name] for name in sorted(RENDERERS)},
        {name: current[name] for name in sorted(RENDERERS)},
    ]


def validate_terminal(terminal, accepted, regression_sha256):
    require(
        regression_sha256 == REGRESSION_SHA, "Exact actual CPU979537 regression receipt required"
    )
    require(
        terminal["status"] == "actual_missing_only3887_exact_FP32_context_alias_recovery_accepted"
        and terminal["original_context_job"] == "978414"
        and terminal["original_raw_job"] == "978318"
        and str(terminal["successor_raw_job"]).isdecimal()
        and terminal["studies"] == 3887
        and terminal["exact_both_view_images_targets_metadata_priors"] is True
        and terminal["global_RNG_unchanged"] is True
        and terminal["all_originals_preserved"] is True,
        "Exact complete alias-recovery terminal required",
    )
    require(
        terminal["original_renderer_sha256"] == accepted[0]
        and terminal["corrected_renderer_sha256"] == accepted[1]
        and terminal["alias_regression_receipt_sha256"] == regression_sha256
        and terminal["prefix_admission_receipt_sha256"] == PREFIX_SHA,
        "Terminal old/new renderer or regression binding differs",
    )


def renderer_counts(rows, accepted):
    counts = {"original": 0, "corrected": 0}
    for row in rows:
        actual = row["provenance"]["module_sha256"]
        require(actual in accepted, "Unexpected per-row renderer provenance")
        counts["original" if actual == accepted[0] else "corrected"] += 1
    return counts


def validate_owner_stop(receipt):
    require(
        receipt["status"]
        == "obsolete_local_waiting_owner_stopped_before_any_mutation_for_source_compatibility_successor"
        and receipt["PID"] == 59538
        and receipt["owner_git"] == "307a28b79dddb38a4a61cb997422cd5b30e12dc1"
        and receipt["science_git"] == ORIGINAL_SCIENCE
        and set(receipt["journal_members"])
        == {"launcher.json", "owner-intent.json", "stderr", "stdout"}
        and receipt["phase_intents"] == 0
        and receipt["transport_children"] == 0
        and receipt["local_owner_exit_confirmed"] is True
        and receipt["remote_jobs_cancelled"] == 0
        and receipt["existing_originals_modified"] is False,
        "Actual pre-mutation old-owner stop receipt required",
    )
