"""Reuse original members; copy only accepted corrected bodies into new output.

The caller reconciles the original rows first and supplies already resident
authoritative bytes. No network operation is introduced here. The private clone
of the existing resume function avoids changing shared module globals while
multiple study lanes run. All member-level proof fields remain private.
"""

import json
import os
import types
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import source_correction as correction

from rsna.competitive.predictive import source_resume as sr


def _resident(row, resident_paths, changes):
    correction.require(row["path"] in changes and row["path"] in resident_paths,
                       "Unapproved resident correction source")
    after = changes[row["path"]]["after"]
    correction.require(
        all(row[key] == after[key] for key in (*correction.IDENTITY, "bytes", "sha256"))
        and all(row[key] == after[key] for key in after if key in row),
        "Exact effective member required",
    )
    actual = Path(resident_paths[row["path"]])
    signature = sr._signature(actual)
    correction.require(
        signature[2] == after["bytes"] and sr._hash_file(actual) == after["sha256"]
        and sr._signature(actual) == signature,
        "Resident authoritative size/SHA/stat differs",
    )
    return actual, signature


@contextmanager
def resident_response(row, resident_paths, plan_body, receipt_body):
    """Open exact resident corrected bytes with their true corrected revision."""
    changes = correction.validate(plan_body, receipt_body)
    path, signature = _resident(row, resident_paths, changes)
    with path.open("rb") as stream:
        yield sr.SourceResponse(
            stream, 200, {"Content-Length": str(row["bytes"])},
            correction.CORRECTED_REVISION, row["path"],
        )
    correction.require(sr._signature(path) == signature, "Resident correction changed")


def resume(root, original_plan, *, plan_body, receipt_body, resident_paths,
           opener, terminal_receipt, workers=1):
    changes = correction.validate(plan_body, receipt_body)
    correction.require(original_plan.revision == correction.ORIGINAL_REVISION,
                       "Original source revision differs")
    correction.require(set(resident_paths) <= set(changes),
                       "Unapproved resident correction source")
    effective, selected = [], {}
    for member in original_plan.members:
        if member.path not in changes:
            effective.append(member)
            continue
        change = changes[member.path]
        row = member.source_row()
        correction.require(
            all(row[key] == change["before"][key] for key in row)
            and member.path in resident_paths,
            "Exact before member and resident after payload required",
        )
        selected[member.path] = change
        effective.append(replace(
            member, size=change["after"]["bytes"], sha256=change["after"]["sha256"],
            retained_size=0, retained_sha256=None,
        ))
    correction.require(selected, "No approved correction in this original study")
    rows = [member.source_row() for member in effective]
    required = sum(member.size for member in effective) + original_plan.control_reserve
    required += sum(len(body) for _, body in sr._metadata(original_plan.uid, original_plan.series_csv))
    correction.require(required <= original_plan.cap, "Corrected source exceeds original cap")
    plan = replace(
        original_plan, members=tuple(effective),
        source_table_sha256=correction.sha(correction.canonical(rows)),
    )
    resident = {}
    for path, change in selected.items():
        resident[path] = _resident(change["after"], resident_paths, changes)

    @contextmanager
    def open_one(row, offset):
        if row["path"] not in selected:
            with opener(row, offset) as response:
                yield response
            return
        change = selected[row["path"]]
        correction.require(
            offset == 0 and all(row[key] == change["after"][key] for key in row),
            "Only the complete exact authoritative member may replace a defective original",
        )
        path, signature = resident[row["path"]]
        correction.require(sr._signature(path) == signature, "Resident correction changed")
        with path.open("rb") as stream:
            yield sr.SourceResponse(
                stream, 200, {"Content-Length": str(change["after"]["bytes"])},
                correction.CORRECTED_REVISION, row["path"],
            )
        correction.require(sr._signature(path) == signature, "Resident correction changed")

    def accept(response, response_plan, member):
        if member.path in selected:
            change = selected[member.path]
            correction.require(
                member.size == change["after"]["bytes"]
                and member.sha256 == change["after"]["sha256"]
                and member.retained_size == 0,
                "Correction response member differs",
            )
            response_plan = replace(response_plan, revision=correction.CORRECTED_REVISION)
        sr._accept_response(response, response_plan, member)

    scope = dict(sr.resume_missing.__globals__)
    scope["_accept_response"] = accept
    run = types.FunctionType(sr.resume_missing.__code__, scope)
    run.__defaults__, run.__kwdefaults__ = sr.resume_missing.__defaults__, sr.resume_missing.__kwdefaults__
    data, proof = run(
        root, plan, opener=open_one, terminal_receipt=terminal_receipt, workers=workers,
    )
    copied = 0
    for item in proof["files"]:
        if item["source_path"] in selected:
            copied += item["bytes"]
            item.update(action="copied_authoritative_correction", fetched_bytes=0,
                        resident_authoritative_bytes=item["bytes"],
                        source_revision=correction.CORRECTED_REVISION)
        else:
            item["source_revision"] = correction.ORIGINAL_REVISION
    proof.update(
        fetched_bytes=proof["fetched_bytes"] - copied,
        resident_authoritative_bytes=copied,
        original_source_table_sha256=original_plan.source_table_sha256,
        source_correction_plan_sha256=correction.PLAN_SHA,
        source_correction_receipt_sha256=correction.RECEIPT_SHA,
        source_correction_lineage=[{
            "path": path, "before": change["before"], "after": change["after"],
            "original_revision": correction.ORIGINAL_REVISION,
            "effective_revision": correction.CORRECTED_REVISION,
        } for path, change in selected.items()],
        source_revision_is_default_for_unchanged_members=True,
        decoder_changed=False, files_skipped=0,
    )
    body = json.dumps(proof, sort_keys=True, allow_nan=False).encode()
    actual = sum(path.stat().st_size for path in data.rglob("*") if path.is_file())
    correction.require(
        len(body) <= plan.control_reserve and actual + len(body) <= plan.cap,
        "Final corrected source proof exceeds original cap",
    )
    # This receipt belongs to this new output and has not been returned/sealed.
    # Originals and historical proof files are never opened for writing.
    pending = data.parent / "source-proof.corrected.pending"
    with pending.open("xb") as output:
        output.write(body)
        output.flush()
        os.fsync(output.fileno())
    pending.chmod(0o600)
    os.replace(pending, data.parent / "source-proof.private.json")
    return data, proof
