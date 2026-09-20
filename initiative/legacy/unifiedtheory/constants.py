"""Competition constants shared by extraction, training, and inference."""

from __future__ import annotations

TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

PLANE_TO_ID = {
    "unknown": 0,
    "sagittal": 1,
    "coronal": 2,
    "axial": 3,
}

# 0 is unknown, 1 is false, 2 is true.  This distinguishes absent metadata
# from a measured negative value.
TRISTATE_UNKNOWN = 0
TRISTATE_FALSE = 1
TRISTATE_TRUE = 2

CACHE_SCHEMA_VERSION = 1
