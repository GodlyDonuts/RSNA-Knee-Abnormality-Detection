"""Recognize newer registered actors without changing their frozen controllers."""

from types import FunctionType

EXTRA = (
    "family-influence-full:",
    "family-influence-label-recovery:",
    "whole94-fixed-final-qualification-v2:",
    "whole94-fixed-final-public-v2:",
    "finding-strata-recovery:",
)


def admission(base, db, remote):
    function = FunctionType(
        base.admission.__code__,
        {**base.admission.__globals__, "FAMILIES": (*base.FAMILIES, *EXTRA)},
        "expanded_known_campaign_capacity",
    )
    function.__kwdefaults__ = base.admission.__kwdefaults__
    return function(db, remote, caller="sampling")
