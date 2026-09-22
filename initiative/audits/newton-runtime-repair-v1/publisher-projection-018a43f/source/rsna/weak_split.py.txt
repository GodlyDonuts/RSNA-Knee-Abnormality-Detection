"""Deterministic class-coverage repair after fixed weak-supervision masking.

Only split values can change. Official gold values are never inspected; gold and
quarantine groups are never candidates. This utility does not choose masks.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import pandas as pd

from rsna.schema import TARGETS

SPLIT_REPAIR_VERSION = "whole_group_post_mask_class_coverage_v1"
ACTIVE_SPLITS = ("train", "weak_validation")
MAX_MOVES = 2 * len(TARGETS) * 2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("Weak split repair: " + message)


def _validated(studies: pd.DataFrame) -> tuple[dict, dict, dict]:
    required = {"study_uid", "leakage_group", "split"}
    required.update(TARGETS)
    required.update(f"{target}__mask" for target in TARGETS)
    _require(
        studies.columns.is_unique and required <= set(studies.columns),
        "missing/duplicate columns",
    )
    _require(
        not studies.empty and not studies.study_uid.duplicated().any(),
        "empty/duplicate studies",
    )
    for column in ("study_uid", "leakage_group"):
        _require(
            studies[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all(),
            "invalid " + column,
        )
    _require(
        studies.split.isin([*ACTIVE_SPLITS, "gold_evaluation", "quarantine"]).all(),
        "unknown split",
    )
    groups = {}
    group_classes = {}
    counts: dict[str, Counter[tuple[str, int]]] = {split: Counter() for split in ACTIVE_SPLITS}
    active = studies.split.isin(ACTIVE_SPLITS)
    for target in TARGETS:
        mask = studies[target + "__mask"]
        _require(mask.isin([0, 1]).all(), "invalid mask for " + target)
        _require(
            studies.loc[active & mask.eq(1), target].isin([0, 1]).all(),
            "nonbinary observed weak value for " + target,
        )
    for group, rows in studies.groupby("leakage_group", sort=True):
        splits = set(rows.split)
        if splits & set(ACTIVE_SPLITS):
            _require(len(splits) == 1, "leakage group crosses an active split boundary")
        else:
            continue
        split = next(iter(splits))
        groups[group] = {
            "split": split,
            "uids": tuple(sorted(rows.study_uid)),
            "size": len(rows),
        }
        classes: Counter[tuple[str, int]] = Counter()
        for target in TARGETS:
            observed = rows.loc[rows[target + "__mask"].eq(1), target]
            for value in (0, 1):
                classes[target, value] = int(observed.eq(value).sum())
        group_classes[group] = classes
        counts[split].update(classes)
    return groups, group_classes, counts


def split_class_counts(studies: pd.DataFrame) -> list[dict]:
    """Weak classes only; never compute gold-class diagnostics."""
    _, _, counts = _validated(studies)
    return [
        {
            "split": split,
            "target": target,
            "negative": counts[split][target, 0],
            "positive": counts[split][target, 1],
        }
        for split in ACTIVE_SPLITS
        for target in TARGETS
    ]


def repair_weak_split_coverage(studies: pd.DataFrame) -> pd.DataFrame:
    """Fill missing weak split classes with minimal safe whole-group moves.

    The original split is retained if complete. Each move increases the number
    of present split/target/classes, preserves all classes already present in the
    donor, and moves every study of one leakage group. Candidate ordering is
    group study count, SHA256 of the existing leakage-group identifier, then that
    identifier. Missing-pair order is train, weak_validation, canonical TARGETS,
    negative then positive. No class or gold metric selects semantic membership.
    """
    groups, group_classes, counts = _validated(studies)
    pairs = [(target, value) for target in TARGETS for value in (0, 1)]
    _require(
        all(sum(counts[s][pair] for s in ACTIVE_SPLITS) > 0 for pair in pairs),
        "target/class globally absent from usable weak supervision",
    )
    output = studies.copy(deep=True)
    for _ in range(MAX_MOVES):
        missing = [
            (split, pair) for split in ACTIVE_SPLITS for pair in pairs if counts[split][pair] == 0
        ]
        if not missing:
            return output
        covered_before = 2 * len(pairs) - len(missing)
        selected = None
        # Try all missing classes in fixed order. A different safe move can make
        # progress when the first missing class has no immediately safe donor.
        for recipient, needed in missing:
            donor = ACTIVE_SPLITS[1] if recipient == ACTIVE_SPLITS[0] else ACTIVE_SPLITS[0]
            candidates = [
                group
                for group, info in groups.items()
                if info["split"] == donor
                and group_classes[group][needed] > 0
                and all(
                    counts[donor][pair] - group_classes[group][pair] > 0
                    for pair in pairs
                    if counts[donor][pair] > 0
                )
            ]
            if candidates:
                group = min(
                    candidates,
                    key=lambda g: (
                        groups[g]["size"],
                        hashlib.sha256(g.encode()).hexdigest(),
                        g,
                    ),
                )
                selected = group, donor, recipient
                break
        if selected is None:
            raise ValueError(
                "Weak split repair: no safe whole leakage-group move can improve missing class coverage"
            )
        group, donor, recipient = selected
        classes = group_classes[group]
        for pair in pairs:
            counts[donor][pair] -= classes[pair]
            counts[recipient][pair] += classes[pair]
        groups[group]["split"] = recipient
        output.loc[output.leakage_group.eq(group), "split"] = recipient
        covered_after = sum(counts[s][pair] > 0 for s in ACTIVE_SPLITS for pair in pairs)
        _require(covered_after > covered_before, "coverage did not improve monotonically")
    _require(
        all(counts[s][pair] > 0 for s in ACTIVE_SPLITS for pair in pairs),
        "move limit exceeded",
    )
    return output


def split_repair_audit(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Serialize exact actual moves independently from the repair's internal loop."""
    _require(
        before.index.equals(after.index) and before.columns.equals(after.columns),
        "row/column structure changed",
    )
    pd.testing.assert_frame_equal(
        before.drop(columns="split"), after.drop(columns="split"), check_exact=True
    )
    _validated(before)
    _validated(after)
    changed = before.split.ne(after.split)
    records = []
    for group in sorted(set(before.loc[changed, "leakage_group"])):
        left = before.loc[before.leakage_group.eq(group)]
        right = after.loc[after.leakage_group.eq(group)]
        _require(len(set(left.split)) == len(set(right.split)) == 1, "partial group move")
        source, destination = left.split.iloc[0], right.split.iloc[0]
        _require(
            source in ACTIVE_SPLITS and destination in ACTIVE_SPLITS and source != destination,
            "protected split changed",
        )
        records.append(
            {
                "leakage_group": group,
                "study_uids": sorted(left.study_uid),
                "study_count": len(left),
                "from_split": source,
                "to_split": destination,
                "tie_break_sha256": hashlib.sha256(group.encode()).hexdigest(),
            }
        )
    return {
        "rule_version": SPLIT_REPAIR_VERSION,
        "moved_groups": records,
        "moved_studies": int(changed.sum()),
        "before_counts": split_class_counts(before),
        "after_counts": split_class_counts(after),
        "only_split_values_changed": True,
        "gold_quarantine_groups_unchanged": True,
        "mask_and_clinical_fields_unchanged": True,
    }
