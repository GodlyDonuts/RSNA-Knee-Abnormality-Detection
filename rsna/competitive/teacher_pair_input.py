"""Bind a privately verified final H100 model to one private Kaggle input.

Pure metadata checks only; does not accept a model graph, create a Dataset,
qualify T4 execution or submit a competition entry. Those have separate receipts.
"""

from __future__ import annotations

import re

SOURCE = "7f75fd46b5d0e772fd2767400e81d70b593619b4"
GPU = "975297"
SHA = re.compile(r"[0-9a-f]{64}")


def candidate(compiled, publication, role):
    sha = publication.get("model_sha256")
    accepted = compiled.get("accepted_snapshot", {})
    if (
        role not in ("pointwise", "teacher_rank")
        or compiled.get("status") != "actual_H100_pair_fixed_final_serving_sources_compiled"
        or compiled.get("training_source_git") != SOURCE
        or compiled.get("role") != role
        or compiled.get("epoch") != 3
        or compiled.get("model_sha256") != sha
        or compiled.get("new_GPU_Dataset_or_competition_mutations") != 0
        or compiled.get("actual_new_T4_serving_accepted") is not False
        or publication.get("status")
        != "actual_teacher_pair_immutable_epoch_private_all_objects_download_size_SHA_verified"
        or publication.get("GPU_job") != GPU
        or publication.get("source_git") != SOURCE
        or publication.get("role") != role
        or publication.get("epoch") != 3
        or publication.get("completed_updates") != 1458
        or publication.get("objects") != 6
        or publication.get("canonical_shared_publication_lock_used") is not True
        or publication.get("prefix")
        != f"checkpoints/teacher-pair-full3887-{SOURCE[:12]}/{role}/epoch-003"
        or not isinstance(sha, str)
        or not SHA.fullmatch(sha)
        or type(publication.get("model_bytes")) is not int
        or not 0 < publication["model_bytes"] < 330 * 2**20
        or accepted.get("role") != role
        or accepted.get("epoch") != 3
        or accepted.get("completed") != 1458
        or accepted.get("full_population") != 3887
        or accepted.get("training_Slurm_job") != GPU
        or accepted.get("training_world_size") != 1
        or accepted.get("anatomical_views_per_study") != 2
        or accepted.get("training_windows") != 16
        or accepted.get("serving_windows") != 94
        or accepted.get("model_sha256") != sha
        or accepted.get("actual_new_T4_acceptance") is not False
    ):
        raise ValueError(
            "Original fixed-final privately verified H100 model and actual CPU compilation required"
        )
    return {
        "role": role,
        "epoch": 3,
        "completed_updates": 1458,
        "training_source_git": SOURCE,
        "training_Slurm_job": GPU,
        "model_bytes": publication["model_bytes"],
        "model_sha256": sha,
        "slug": "godlydonuts/rsna-pair-"
        + ("point" if role == "pointwise" else "rank")
        + "-e3-"
        + sha[:12],
        "ledger_key": "teacher-pair-full-final-private-input:" + role + ":" + sha,
        "private_required": True,
        "version_required": 1,
        "competition_entry_or_T4_acceptance": False,
    }
