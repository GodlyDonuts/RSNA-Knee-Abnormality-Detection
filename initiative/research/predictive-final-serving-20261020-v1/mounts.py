"""Resolve exact Kaggle input mounts without walking unrelated MRI trees."""

import re
from pathlib import Path

PARENT_DATASET = "nartaa/rsna-knee-publication-swa-weights-20261007"
PARENT_FILENAME = "raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt"
COMPETITION = "rsna-knee-abnormality-detection"


def dataset_parts(reference):
    if not isinstance(reference, str):
        raise ValueError("Exact owner/slug Dataset reference required")
    parts = reference.split("/")
    if len(parts) != 2 or any(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", p) is None for p in parts):
        raise ValueError("Exact owner/slug Dataset reference required")
    return tuple(parts)


def _unique(candidates):
    # Flat and namespaced aliases may point to the same mounted file.
    found = {p.resolve(): p for p in candidates if p.is_file()}
    if len(found) != 1:
        raise ValueError("Exactly one file in the declared Kaggle mount required")
    return next(iter(found.values()))


def resolve_dataset_file(root, reference, filename):
    owner, slug = dataset_parts(reference)
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename in (".", "..")
        or "\\" in filename
    ):
        raise ValueError("One literal Dataset filename required")
    root = Path(root)
    return _unique((root / slug / filename, root / "datasets" / owner / slug / filename))


def resolve_competition_roster(root):
    root = Path(root)
    directories = (root / COMPETITION, root / "competitions" / COMPETITION)
    return _unique([p / "test.csv" for p in directories if (p / "test_series").is_dir()])
