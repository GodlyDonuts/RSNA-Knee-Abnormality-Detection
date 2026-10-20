"""Derive a missing-only HTTP499 continuation; decoder/renderer bytes stay frozen."""

NAME = "predictive-cache-http499-recovery-20261020-v1"
ROOTS = {r: f"/tmp/rsna-predictive-full-{r}-http499-recovery-20261020-v1" for r in ("raw", "context")}
PREVIOUS_ROOTS = {r: f"/tmp/rsna-predictive-full-{r}-data-recovery-20261020-v1" for r in ROOTS}
STATUS = "actual_mixed_corrected_five_generation_HTTP499_prefix_read_only_admitted"


def replace(source, old, new, count=1):
    if source.count(old) != count:
        raise ValueError(f"Frozen operational anchor differs: {old[:100]}")
    return source.replace(old, new)


def common(source):
    for role in ROOTS:
        source = source.replace(PREVIOUS_ROOTS[role], ROOTS[role])
    return replace(source, '"continuation_chain": continuation_chain, **correction_meta,',
                   '"continuation_chain": continuation_chain, **correction_meta,\n'
                   '            **suffix.http499_metadata(v),', 2)


def configure_before(source, lhs):
    anchor = lhs + " = suffix.raw_inventory"
    return replace(source, anchor,
        'suffix.configure(v, records, location["sha256"])\n' + anchor)


def raw(source):
    source = common(source)
    anchor = "http_predecessor = Path("
    end = source.index("\n", source.index(anchor))
    source = source[:end] + f'\ndata_predecessor = Path({PREVIOUS_ROOTS["raw"]!r})' + source[end:]
    source = replace(source, "[original, predecessor, capacity_predecessor, http_predecessor]",
                     "[original, predecessor, capacity_predecessor, http_predecessor, data_predecessor]")
    source = configure_before(source, "inv")
    source = replace(source, '        + list((http_predecessor / "cache").glob("*.npz"))',
        '        + list((http_predecessor / "cache").glob("*.npz"))\n'
        '        + list((data_predecessor / "cache").glob("*.npz"))')
    needle = ' + list(Path("' + ROOTS['context'] + '").glob("context-*.npz"))'
    source = replace(source, needle,
        ' + list(Path("' + PREVIOUS_ROOTS['context'] + '").glob("context-*.npz"))' + needle)
    source = source.replace('http_predecessor / "scratch"',
                             'http_predecessor / "scratch", data_predecessor / "scratch"')
    source = source.replace('control_bytes(http_predecessor) + control_bytes(D)',
        'control_bytes(http_predecessor) + control_bytes(data_predecessor) + control_bytes(D)')
    return source


def context(source):
    source = common(source)
    anchor = 'http_context = Path('
    end = source.index('\n', source.index(anchor))
    source = source[:end] + (f'\ndata_raw = Path({PREVIOUS_ROOTS["raw"]!r})\n'
                            f'data_context = Path({PREVIOUS_ROOTS["context"]!r})') + source[end:]
    source = replace(source, '[old_raw, predecessor_raw, capacity_raw, http_raw]',
                     '[old_raw, predecessor_raw, capacity_raw, http_raw, data_raw]')
    source = replace(source, '[old_context, predecessor_context, capacity_context, http_context]',
                     '[old_context, predecessor_context, capacity_context, http_context, data_context]')
    source = configure_before(source, 'original')
    source = replace(source, '        http_raw / "cache" / f"case-{i:04d}.private.npz",',
        '        http_raw / "cache" / f"case-{i:04d}.private.npz",\n'
        '        data_raw / "cache" / f"case-{i:04d}.private.npz",')
    source = source.replace('(old_raw, predecessor_raw, capacity_raw, http_raw, new_raw)',
                             '(old_raw, predecessor_raw, capacity_raw, http_raw, data_raw, new_raw)')
    source = source.replace('(old_context, predecessor_context, capacity_context, http_context, D)',
        '(old_context, predecessor_context, capacity_context, http_context, data_context, D)')
    source = source.replace('controls_size(http_context) + controls_size(D)',
        'controls_size(http_context) + controls_size(data_context) + controls_size(D)')
    source = replace(source, 'assert raw_intent["source_http_sha256"] == v["source_http_sha256"]',
        'assert all(raw_intent[k] == x for k, x in suffix.http499_metadata(v).items())')
    return source


def admit(source):
    for role in ROOTS:
        source = replace(source, f'Path(chain["{role}"]["root"])]',
            f'Path("/tmp/rsna-predictive-full-{role}-http-recovery-20261020-v1"), '
            f'Path(chain["{role}"]["root"])]')
        source = source.replace(PREVIOUS_ROOTS[role], ROOTS[role])
    source = configure_before(source, 'raw')
    source = replace(source, repr("actual_mixed_HTTP_prefix_and_authoritative_overlay_read_only_admitted"), repr(STATUS))
    source = replace(source, '"original_jobs": suffix.PREDECESSORS',
                     '"original_jobs": suffix.HTTP499_PREDECESSORS')
    source = replace(source, '"module_sha256": original_renderer if original else renderer}',
        '"module_sha256": original_renderer if original else renderer,\n'
        '            **({"source_correction": row["source_correction"]} if "source_correction" in row else {})}')
    source = replace(source,
        'assert not (set(corrections) & (set(contexts["complete"]) | set(contexts["orphan_caches"])))',
        'suffix.require_corrected_contexts(contexts, corrections)')
    source = replace(source, 'print(json.dumps({**correction_meta,',
        'print(json.dumps({**correction_meta, **suffix.http499_metadata(v),')
    return source


def programs(original):
    result = {"raw": raw(original["raw"]), "context": context(original["context"]),
              "admit": admit(original["admit"]), "collect": original["collect"]}
    for role, text in result.items():
        compile(text, f"exact_HTTP499_{role}", "exec")
    return result
