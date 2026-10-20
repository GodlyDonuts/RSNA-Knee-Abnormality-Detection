"""Derive one explicit transport successor from exact frozen capacity controls."""

OLD_NAME = "predictive-cache-capacity-recovery-20261020-v1"
NAME = "predictive-cache-http-recovery-20261020-v1"
ROOTS = {
    role: f"/tmp/rsna-predictive-full-{role}-http-recovery-20261020-v1"
    for role in ("raw", "context")
}
PREVIOUS_ROOTS = {
    role: f"/tmp/rsna-predictive-full-{role}-capacity-recovery-20261020-v1"
    for role in ("raw", "context")
}


def replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f"Frozen operational anchor count differs: {old[:90]}")
    return source.replace(old, new)


def common(source):
    # Keep the existing alias ancestry/status; add this separate HTTP generation.
    source = source.replace(PREVIOUS_ROOTS["raw"], ROOTS["raw"])
    source = source.replace(PREVIOUS_ROOTS["context"], ROOTS["context"])
    source = replace(
        source,
        '"mixed_prefix_admission_sha256": v["mixed_prefix_admission_sha256"],',
        '"mixed_prefix_admission_sha256": v["mixed_prefix_admission_sha256"],\n'
        '            "http_recovery_decision_sha256": v["http_recovery_decision_sha256"],\n'
        '            "source_http_sha256": v["source_http_sha256"],',
        count=2,
    )
    return source


def raw(source):
    source = common(source)
    source = replace(
        source,
        'predecessor = Path("/tmp/rsna-predictive-full-raw-alias-recovery-20261020-v3")',
        'predecessor = Path("/tmp/rsna-predictive-full-raw-alias-recovery-20261020-v3")\n'
        f'capacity_predecessor = Path({PREVIOUS_ROOTS["raw"]!r})',
    )
    source = replace(source, "[original, predecessor]", "[original, predecessor, capacity_predecessor]")
    source = replace(
        source, '    "rsna/competitive/predictive/source_resume.py",',
        '    "rsna/competitive/predictive/source_resume.py",\n'
        '    "rsna/competitive/predictive/source_http.py",',
    )
    source = replace(
        source, '        + list((predecessor / "cache").glob("*.npz"))',
        '        + list((predecessor / "cache").glob("*.npz"))\n'
        '        + list((capacity_predecessor / "cache").glob("*.npz"))',
    )
    needle = ' + list(Path("' + ROOTS["context"] + '").glob("context-*.npz"))'
    source = replace(
        source, needle,
        ' + list(Path("' + PREVIOUS_ROOTS["context"] + '").glob("context-*.npz"))' + needle,
    )
    source = source.replace(
        '[original / "scratch", predecessor / "scratch"',
        '[original / "scratch", predecessor / "scratch", capacity_predecessor / "scratch"',
    )
    source = source.replace(
        "control_bytes(original) + control_bytes(predecessor) + control_bytes(D)",
        "control_bytes(original) + control_bytes(predecessor) + "
        "control_bytes(capacity_predecessor) + control_bytes(D)",
    )
    source = replace(
        source, "from rsna.competitive.predictive import source_resume",
        "from rsna.competitive.predictive import source_resume\n"
        "from rsna.competitive.predictive.source_http import SourceHTTP",
    )
    source = replace(
        source, 'token = v.pop("token")',
        'token = v.pop("token")\n'
        'http = SourceHTTP(revision=revision, token=token,\n'
        '                  emit=lambda event: print(json.dumps({"source_HTTP": event}), flush=True))',
    )
    start = source.index("@contextlib.contextmanager\ndef range_opener(")
    end = source.index("\n\n@contextlib.contextmanager\ndef opener(", start)
    source = source[:start] + (
        "@contextlib.contextmanager\n"
        "def range_opener(row, offset):\n"
        "    with http.open_response(row, offset) as response:\n"
        "        yield response\n"
    ) + source[end:]
    source = replace(
        source,
        'terminal_receipt=(terminal if inv["buffer_owners"][str(inv["buffers"][i])] == original\n'
        '                              else continuation_chain["raw"]["actual_terminal"])',
        'terminal_receipt=(terminal if inv["buffer_owners"][str(inv["buffers"][i])] == original\n'
        '                else continuation_chain["raw"]["ancestor_chain"]["raw"]["actual_terminal"]\n'
        '                if inv["buffer_owners"][str(inv["buffers"][i])] == predecessor\n'
        '                else continuation_chain["raw"]["actual_terminal"])',
    )
    return source


def context(source):
    source = common(source)
    source = replace(
        source, 'predecessor_raw = Path("/tmp/rsna-predictive-full-raw-alias-recovery-20261020-v3")',
        'predecessor_raw = Path("/tmp/rsna-predictive-full-raw-alias-recovery-20261020-v3")\n'
        f'capacity_raw = Path({PREVIOUS_ROOTS["raw"]!r})\n'
        f'capacity_context = Path({PREVIOUS_ROOTS["context"]!r})',
    )
    source = replace(source, "[old_raw, predecessor_raw]", "[old_raw, predecessor_raw, capacity_raw]")
    source = replace(source, "[old_context, predecessor_context]",
                     "[old_context, predecessor_context, capacity_context]")
    source = replace(
        source, '        predecessor_raw / "cache" / f"case-{i:04d}.private.npz",',
        '        predecessor_raw / "cache" / f"case-{i:04d}.private.npz",\n'
        '        capacity_raw / "cache" / f"case-{i:04d}.private.npz",',
    )
    source = source.replace("(old_raw, predecessor_raw, new_raw)",
                            "(old_raw, predecessor_raw, capacity_raw, new_raw)")
    source = source.replace("(old_context, predecessor_context, D)",
                            "(old_context, predecessor_context, capacity_context, D)")
    source = source.replace(
        "controls_size(old_context) + controls_size(predecessor_context) + controls_size(D)",
        "controls_size(old_context) + controls_size(predecessor_context) + "
        "controls_size(capacity_context) + controls_size(D)",
    )
    source = replace(
        source,
        'assert raw_intent["capacity_decision_sha256"] == v["capacity_decision_sha256"]',
        'assert raw_intent["capacity_decision_sha256"] == v["capacity_decision_sha256"]\n'
        'assert raw_intent["http_recovery_decision_sha256"] == v["http_recovery_decision_sha256"]\n'
        'assert raw_intent["source_http_sha256"] == v["source_http_sha256"]',
    )
    return source


def admit(source):
    for role, stem in (("raw", "rawroots"), ("context", "ctxroots")):
        old = f'{stem} = [Path("/tmp/rsna-predictive-full-{role}-cache-20261020-v1"), Path(chain["{role}"]["root"])]'
        new = (
            f'{stem} = [Path("/tmp/rsna-predictive-full-{role}-cache-20261020-v1"), '
            f'Path("/tmp/rsna-predictive-full-{role}-alias-recovery-20261020-v3"), '
            f'Path(chain["{role}"]["root"])]'
        )
        source = replace(source, old, new)
    source = replace(source, '"actual_mixed_v3_prefix_read_only_admitted"',
                     '"actual_mixed_v4_http_prefix_read_only_admitted"')
    source = replace(
        source, '    "capacity_decision_sha256": v["capacity_decision_sha256"],',
        '    "capacity_decision_sha256": v["capacity_decision_sha256"],\n'
        '    "http_recovery_decision_sha256": v["http_recovery_decision_sha256"],\n'
        '    "source_http_sha256": v["source_http_sha256"],',
    )
    source = replace(
        source, "index = rawroots[0]",
        f'assert not Path({ROOTS["raw"]!r}).exists() and not Path({ROOTS["context"]!r}).exists()\n'
        "index = rawroots[0]",
    )
    return source


def programs(original):
    result = {"raw": raw(original["raw"]), "context": context(original["context"]),
              "admit": admit(original["admit"]), "collect": original["collect"]}
    for role, source in result.items():
        compile(source, f"distinct_http_recovery_{role}", "exec")
    return result
