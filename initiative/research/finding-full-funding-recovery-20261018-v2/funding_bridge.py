"""Replace only retired completed-family funding; preserve all numerical code."""

from __future__ import annotations

import hashlib

PINS = {
    "full": "d76b3c7225d96dcf647039159472e0e4e269ad88efe89c7294b8075a78740da1",
    "publication": "4cfeb1f799991377df2ee31a4b26ea8ebe0b16c4f9ffe5bcc19e1b128057e5a1",
}
RETIREMENT_SHA = "f210549eb1d2aa76a62140ce8bbc9ae03164f1d68542f191357710ee0a00c54a"


def adapt(source, kind):
    if kind not in PINS or hashlib.sha256(source.encode()).hexdigest() != PINS[kind]:
        raise ValueError("Exact frozen original admission/publication source required")
    if kind == "full":
        old = "    assert shutil.disk_usage('/tmp').free>603433009211+max(0,3*2**30-paid)+max(0,4*2**30-family_paid)"
        new = (
            "    retirement=family_storage/'completed-fixed3-unspent512MiB-retirement-v1.json';assert hashlib.sha256(retirement.read_bytes()).hexdigest()=="
            + repr(RETIREMENT_SHA)
            + ";cap=json.loads(retirement.read_text());assert cap['new_family_cap_bytes']==3758096384 and cap['retired_previous_N3_eightGiB_again'] is False and cap['physical_deletion_or_free_space_credit']==0\n    assert family_paid<cap['new_family_cap_bytes'] and shutil.disk_usage('/tmp').free>603433009211+max(0,3*2**30-paid)+max(0,cap['new_family_cap_bytes']-family_paid)"
        )
    else:
        old = "assert shutil.disk_usage('/tmp').free>603433009211+max(0,3*2**30-paid)+max(0,4*2**30-family_paid)"
        new = (
            "retirement=F/'completed-fixed3-unspent512MiB-retirement-v1.json';assert hashlib.sha256(retirement.read_bytes()).hexdigest()=="
            + repr(RETIREMENT_SHA)
            + ";cap=json.loads(retirement.read_text());assert cap['new_family_cap_bytes']==3758096384 and cap['retired_previous_N3_eightGiB_again'] is False and cap['physical_deletion_or_free_space_credit']==0\nassert family_paid<cap['new_family_cap_bytes'] and shutil.disk_usage('/tmp').free>603433009211+max(0,3*2**30-paid)+max(0,cap['new_family_cap_bytes']-family_paid)"
        )
    if source.count(old) != 1:
        raise ValueError("Unique original resource-only guard required")
    changed = source.replace(old, new)
    if changed.replace(new, old) != source:
        raise ValueError("Funding-only replacement required")
    compile(changed, "original_science_retired_future_capacity_only", "exec")
    return changed
