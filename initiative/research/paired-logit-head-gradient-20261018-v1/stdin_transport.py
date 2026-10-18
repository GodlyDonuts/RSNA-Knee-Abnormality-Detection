"""Bounded SSH command; scientific program and private payload stay in stdin."""

import json

COMMAND = (
    "import io,json,sys; e=json.load(sys.stdin); "
    "sys.stdin=io.StringIO(json.dumps(e['payload'],allow_nan=False)); "
    "exec(compile(e['program'],'original_frozen_scientific_program','exec'),"
    "{'__name__':'__main__'})"
)


def envelope(program, payload):
    if not isinstance(program, str) or not program or not isinstance(payload, dict):
        raise ValueError("Explicit frozen program and structured stdin payload required")
    compile(program, "frozen_program_before_transport", "exec")
    return json.dumps({"program": program, "payload": payload}, ensure_ascii=False, allow_nan=False)
