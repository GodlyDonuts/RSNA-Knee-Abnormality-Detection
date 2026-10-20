"""Exact operational derivation; original decoder and renderer remain unchanged."""

NAME = "predictive-cache-data-recovery-20261020-v1"
ROOTS = {r: f"/tmp/rsna-predictive-full-{r}-data-recovery-20261020-v1" for r in ("raw", "context")}
PREVIOUS_ROOTS = {r: f"/tmp/rsna-predictive-full-{r}-http-recovery-20261020-v1" for r in ROOTS}
STATUS = "actual_mixed_HTTP_prefix_and_authoritative_overlay_read_only_admitted"


def replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f"Exact frozen operational anchor differs: {old[:100]}")
    return source.replace(old, new)


def common(source):
    for role in ROOTS:
        source = source.replace(PREVIOUS_ROOTS[role], ROOTS[role])
    return replace(source, '"continuation_chain": continuation_chain,',
                   '"continuation_chain": continuation_chain, **correction_meta,', 2)


def raw(source):
    source = common(source)
    anchor = 'capacity_predecessor = Path('
    end = source.index('\n', source.index(anchor))
    source = source[:end] + f'\nhttp_predecessor = Path({PREVIOUS_ROOTS["raw"]!r})' + source[end:]
    source = replace(source, '[original, predecessor, capacity_predecessor]',
                     '[original, predecessor, capacity_predecessor, http_predecessor]')
    source = replace(source, 'suffix.check_admission(v, inv, recon)',
        'suffix.check_admission(v, inv, recon)\noriginal_records = records\n'
        'records, corrections, correction_meta, resident_paths = suffix.overlay(v, records, location["sha256"])\n'
        'suffix.require_unsealed(inv, corrections)')
    source = replace(source, '        + list((capacity_predecessor / "cache").glob("*.npz"))',
        '        + list((capacity_predecessor / "cache").glob("*.npz"))\n'
        '        + list((http_predecessor / "cache").glob("*.npz"))')
    needle = ' + list(Path("' + ROOTS['context'] + '").glob("context-*.npz"))'
    source = replace(source, needle,
        ' + list(Path("' + PREVIOUS_ROOTS['context'] + '").glob("context-*.npz"))' + needle)
    source = source.replace('capacity_predecessor / "scratch"',
                             'capacity_predecessor / "scratch", http_predecessor / "scratch"')
    source = source.replace('control_bytes(capacity_predecessor) + control_bytes(D)',
                             'control_bytes(capacity_predecessor) + control_bytes(http_predecessor) + control_bytes(D)')
    source = replace(source, '        c = records[i]', '        c = original_records[i]')
    source = replace(source, 'uid_to_index = {row["uid"]: i for i, row in enumerate(records)}',
        'uid_to_index = {row["uid"]: i for i, row in enumerate(records)}\n'
        'overlay_resume = suffix.overlay_module(v)')
    source = replace(source, '    with http.open_response(row, offset) as response:\n        yield response',
        '    if row["path"] in resident_paths:\n'
        '        assert offset == 0\n'
        '        with overlay_resume.resident_response(row, resident_paths=resident_paths,\n'
        '                plan_body=v["source_correction_plan_body"].encode(),\n'
        '                receipt_body=v["source_correction_receipt_body"].encode()) as response:\n'
        '            yield response\n'
        '    else:\n'
        '        with http.open_response(row, offset) as response:\n'
        '            yield response')
    begin = source.index('        return source_resume.resume_missing(')
    end = source.index('\n    return old_helper["acquire"]', begin)
    source = source[:begin] + '''        terminal_proof = suffix.buffer_terminal(continuation_chain, inv["buffer_owners"][str(inv["buffers"][i])], terminal)
        if i in corrections:
            return overlay_resume.resume(root, plans[i],
                plan_body=v["source_correction_plan_body"].encode(),
                receipt_body=v["source_correction_receipt_body"].encode(),
                resident_paths=resident_paths, opener=range_opener,
                terminal_receipt=terminal_proof, workers=workers)
        return source_resume.resume_missing(root, plans[i], opener=range_opener,
            terminal_receipt=terminal_proof, workers=workers)''' + source[end:]
    source = replace(source, 'start = time.monotonic()', '''# Only source provenance is extended; the original decoder/render body is intact.
one_text = suffix.extend_one(one_text)
start = time.monotonic()''')
    return source


def context(source):
    source = common(source)
    anchor = 'capacity_context = Path('
    end = source.index('\n', source.index(anchor))
    source = source[:end] + (f'\nhttp_raw = Path({PREVIOUS_ROOTS["raw"]!r})\n'
                             f'http_context = Path({PREVIOUS_ROOTS["context"]!r})') + source[end:]
    source = replace(source, '[old_raw, predecessor_raw, capacity_raw]',
                     '[old_raw, predecessor_raw, capacity_raw, http_raw]')
    source = replace(source, '[old_context, predecessor_context, capacity_context]',
                     '[old_context, predecessor_context, capacity_context, http_context]')
    source = replace(source, 'suffix.check_admission(v, original, recon)',
        'suffix.check_admission(v, original, recon)\n'
        'records, corrections, correction_meta, _ = suffix.overlay(v, records, location["sha256"])\n'
        'suffix.require_unsealed(original, corrections)\n'
        'assert all(raw_intent[k] == value for k, value in correction_meta.items())')
    source = replace(source, '        "module_sha256": original_renderer if original else renderer,',
        '        "module_sha256": original_renderer if original else renderer,\n'
        '        **({"source_correction": row["source_correction"]} if "source_correction" in row else {}),')
    source = replace(source, '        capacity_raw / "cache" / f"case-{i:04d}.private.npz",',
        '        capacity_raw / "cache" / f"case-{i:04d}.private.npz",\n'
        '        http_raw / "cache" / f"case-{i:04d}.private.npz",')
    source = source.replace('(old_raw, predecessor_raw, capacity_raw, new_raw)',
                             '(old_raw, predecessor_raw, capacity_raw, http_raw, new_raw)')
    source = source.replace('(old_context, predecessor_context, capacity_context, D)',
                             '(old_context, predecessor_context, capacity_context, http_context, D)')
    source = source.replace('controls_size(capacity_context) + controls_size(D)',
                             'controls_size(capacity_context) + controls_size(http_context) + controls_size(D)')
    source = replace(source, '    return row\n\n\ndef producer_state',
        '    assert row.get("source_correction") == corrections.get(i)\n'
        '    return row\n\n\ndef producer_state')
    source = replace(source, '''    assert source_proof == recon.source_provenance(
        records[i], "52f995f4568e36bfa5fbbaf69a89af41226b4f5d"
    )''', '''    expected_source = recon.source_provenance(records[i], "52f995f4568e36bfa5fbbaf69a89af41226b4f5d")
    if i in corrections:
        expected_source["source_correction"] = corrections[i]
    assert source_proof == expected_source''')
    return source


def admit(source):
    for role, variable in (("raw", "rawroots"), ("context", "ctxroots")):
        needle = f'Path(chain["{role}"]["root"])]'
        source = replace(source, needle,
            f'Path("/tmp/rsna-predictive-full-{role}-capacity-recovery-20261020-v1"), ' + needle)
    for role in ROOTS:
        source = source.replace(PREVIOUS_ROOTS[role], ROOTS[role])
    source = replace(source, '"actual_mixed_v4_http_prefix_read_only_admitted"', repr(STATUS))
    source = replace(source, 'contexts = suffix.context_inventory',
        'effective, corrections, correction_meta, _ = suffix.overlay(v, records, location["sha256"])\n'
        'suffix.require_unsealed(raw, corrections)\n'
        'contexts = suffix.context_inventory')
    source = replace(source, 'print(json.dumps({',
        'assert not (set(corrections) & (set(contexts["complete"]) | set(contexts["orphan_caches"])))\n'
        'print(json.dumps({**correction_meta,')
    return source


def programs(original):
    result = {"raw": raw(original["raw"]), "context": context(original["context"]),
              "admit": admit(original["admit"]), "collect": original["collect"]}
    for role, text in result.items():
        compile(text, f"exact_source_overlay_{role}", "exec")
    return result
