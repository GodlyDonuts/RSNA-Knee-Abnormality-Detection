"""Independent, local-only validation of the frozen endpoint survey. No DICOM fetches."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3

import pyarrow.dataset as pads

REVISION = '52f995f4568e36bfa5fbbaf69a89af41226b4f5d'
SURVEY_SHA = '206d82db7ddb7a3239119b022fe39150621673899f8757a8553b5ff6fce161ea'
EMPTY_SHA = hashlib.sha256(b'').hexdigest()
FIELDS = ['Laterality', 'ImageLaterality', 'FrameLaterality', 'SeriesDescription', 'SeriesNumber', 'AcquisitionNumber', 'SeriesTime', 'AcquisitionTime', 'PatientPosition', 'ImagePositionPatient', 'ImageOrientationPatient', 'PixelSpacing', 'SliceThickness', 'SpacingBetweenSlices', 'Manufacturer', 'ManufacturerModelName', 'SoftwareVersions', 'Rows', 'Columns', 'PhotometricInterpretation', 'PixelRepresentation', 'RescaleSlope', 'RescaleIntercept']
COLUMNS = ['path', 'study_uid', 'series_uid', 'sop_uid', 'sha256', 'bytes', 'storage']
HASH = re.compile(r'[0-9a-f]{64}\Z')
UID = re.compile(r'[0-9]+(?:\.[0-9]+)+\Z')
SIDE_FIELDS = FIELDS[:3]
PLANE_WORD = re.compile(r'(?<![a-z])(sag(?:ittal)?|cor(?:onal)?|tra(?:ns(?:verse)?)?|ax(?:ial)?)(?![a-z])', re.I)


def check(condition, code):
    if not condition:
        raise ValueError(code)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, value):
    check(not path.exists(), 'immutable_output_already_exists:' + path.name)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def bind(path, base):
    return {'path': str(path.relative_to(base)), 'sha256': file_sha(path), 'bytes': path.stat().st_size}


def inventory_bindings(root):
    return [{'path': str(p.relative_to(root)), 'bytes': p.stat().st_size, 'sha256': file_sha(p)} for p in sorted(root.rglob('*.parquet'))]


def prepare(repo, survey, out):
    manifest = json.loads((survey / 'manifest.json').read_text())
    check(manifest['source_revision'] == REVISION, 'unexpected_source_revision')
    check(manifest['repo_id'] == 'GodlyDonuts/RSNA-Data', 'unexpected_source_repo')
    check(manifest['script_sha256'] == SURVEY_SHA == file_sha(repo.parent / 'header_endpoint_survey.py'), 'survey_code_changed')
    check(manifest['fields'] == FIELDS and manifest['prefix_sizes'] == [65536, 262144, 1048576], 'survey_contract_changed')
    check(manifest['selection'] == {'kind': 'lexical_first_and_last_sop_per_series', 'study_uid': None}, 'not_full_corpus_selection')
    root = repo / 'initiative/source/inventory'
    bindings = inventory_bindings(root)
    check(bindings == manifest['inventory_files'], 'inventory_file_binding_mismatch')
    check(digest(canonical(bindings)) == manifest['inventory_files_fingerprint'], 'inventory_fingerprint_mismatch')
    series = {}
    series_owners = defaultdict(set)
    studies = defaultdict(set)
    split_slices = Counter()
    path_hashes = set()
    n_rows = 0
    for batch in pads.dataset(root, format='parquet').scanner(columns=COLUMNS, batch_size=2048).to_batches():
        for row in batch.to_pylist():
            n_rows += 1
            parts = row['path'].split('/')
            check(len(parts) == 4 and parts[0] in {'train_series', 'test_series'}, 'inventory_path_structure')
            check(parts[1:3] == [row['study_uid'], row['series_uid']] and parts[3] == row['sop_uid'] + '.dcm', 'inventory_path_uid_mismatch')
            check(all(UID.fullmatch(row[k]) and len(row[k]) <= 64 for k in ['study_uid', 'series_uid', 'sop_uid']), 'invalid_inventory_uid')
            check(HASH.fullmatch(row['sha256']) and type(row['bytes']) is int and row['bytes'] > 0 and row['storage'] in {'git', 'lfs'}, 'inventory_object_binding_invalid')
            ph = digest(row['path'].encode())
            check(ph not in path_hashes, 'duplicate_inventory_path')
            path_hashes.add(ph)
            key = (row['study_uid'], row['series_uid'])
            series_owners[row['series_uid']].add(row['study_uid'])
            studies[row['study_uid']].add(parts[0])
            split_slices[parts[0]] += 1
            if key not in series:
                series[key] = [1, row, row]
            else:
                item = series[key]
                item[0] += 1
                if row['sop_uid'] < item[1]['sop_uid']:
                    item[1] = row
                if row['sop_uid'] > item[2]['sop_uid']:
                    item[2] = row
    check(all(len(x) == 1 for x in series_owners.values()), 'series_uid_cross_study_collision')
    check(all(len(x) == 1 for x in studies.values()), 'study_cross_split_collision')
    del path_hashes
    expected = []
    selected = []
    for (study, sid), (n, first, last) in sorted(series.items()):
        selected.append({'study_uid': study, 'series_uid': sid, 'n_slices': n, 'first_sop_uid': first['sop_uid'], 'last_sop_uid': last['sop_uid']})
        expected.append(first)
        if first['sop_uid'] != last['sop_uid']:
            expected.append(last)
    check((n_rows, len(series), len(studies), len(expected)) == (819635, 24386, 4410, 48772), 'unexpected_corpus_counts')
    check(manifest['selected_series'] == selected, 'selected_series_different_from_independent_inventory_scan')
    check(manifest['inventory_slices'] == n_rows and manifest['inventory_series'] == len(series) and manifest['inventory_studies'] == len(studies), 'manifest_inventory_counts')
    check(manifest['n_endpoints'] == len(expected) and manifest['endpoints_sha256'] == digest(canonical(expected)), 'endpoint_selection_digest_mismatch')
    endpoint_file = out / 'expected-endpoints.jsonl'
    check(not endpoint_file.exists(), 'expected_endpoint_file_exists')
    with endpoint_file.open('w') as f:
        for row in expected:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
    result = {'stage': 'independent_inventory_selection_validation', 'agent_id': '/root/image_pipeline_audit', 'source_revision': REVISION, 'manifest_sha256': file_sha(survey / 'manifest.json'), 'validator_sha256': file_sha(Path(__file__)), 'inventory_files': bindings, 'inventory_files_fingerprint': digest(canonical(bindings)), 'inventory_slices': n_rows, 'inventory_series': len(series), 'inventory_studies': len(studies), 'expected_endpoints': len(expected), 'expected_endpoints_sha256': file_sha(endpoint_file), 'endpoint_selection_canonical_sha256': digest(canonical(expected)), 'split_slice_counts': dict(split_slices), 'duplicate_inventory_paths': 0, 'series_uid_cross_study_collisions': 0, 'study_cross_split_collisions': 0, 'scope': 'Independent streaming inventory scan, exact lexical first/last selection and source object binding. No image/header fetches or clinical labels.'}
    dump(out / 'inventory-validation.json', result)
    return result


def validate_record(record, expected):
    key_map = {'source_path': 'path', 'study_uid': 'study_uid', 'series_uid': 'series_uid', 'sop_uid': 'sop_uid', 'source_sha256': 'sha256', 'source_bytes': 'bytes', 'storage': 'storage'}
    for key, col in key_map.items():
        check(record.get(key) == expected[col], 'record_source_binding:' + key)
    required = set(key_map) | {'prefix_bytes', 'prefix_sha256', 'full_file_sha256_verified', 'verification_scope', 'prefix_attempts', 'header_fields', 'patient_id_sha256', 'patient_id_present', 'frame_of_reference_sha256', 'frame_of_reference_present', 'transfer_syntax_uid', 'pixel_data_header', 'reached_pixel_data_header'}
    check(set(record) == required, 'unexpected_record_fields')
    check(type(record['reached_pixel_data_header']) is bool and record['reached_pixel_data_header'], 'pixel_header_not_proven')
    check(set(record['header_fields']) == set(FIELDS) and all(isinstance(v, str) for v in record['header_fields'].values()), 'header_whitelist_mismatch')
    total = record['source_bytes']
    count = record['prefix_bytes']
    budgets = list(dict.fromkeys(min(x, total) for x in [65536, 262144, 1048576]))
    check(type(count) is int and count in budgets, 'invalid_prefix_budget')
    check(HASH.fullmatch(record['prefix_sha256']) is not None, 'invalid_prefix_hash')
    history = record['prefix_attempts']
    check(isinstance(history, list) and history, 'prefix_history_missing')
    check([x['prefix_bytes'] for x in history] == budgets[:len(history)] and history[-1]['prefix_bytes'] == count, 'prefix_history_order')
    check(all(set(x) == {'prefix_bytes', 'prefix_sha256'} and HASH.fullmatch(x['prefix_sha256']) for x in history), 'prefix_history_schema')
    check(history[-1]['prefix_sha256'] == record['prefix_sha256'], 'prefix_history_final_digest')
    full = count == total
    check(type(record['full_file_sha256_verified']) is bool and record['full_file_sha256_verified'] == full, 'full_file_claim_incorrect')
    check(record['verification_scope'] == ('complete_file_sha256' if full else 'range_transport_and_uid_binding_only'), 'verification_scope_incorrect')
    if full:
        check(record['prefix_sha256'] == expected['sha256'], 'full_file_hash_mismatch')
    for kind in ['patient_id', 'frame_of_reference']:
        present = record[kind + '_present']
        h = record[kind + '_sha256']
        check(type(present) is bool and HASH.fullmatch(h), kind + '_hash_schema')
        check(present == (h != EMPTY_SHA), kind + '_empty_hash_inconsistent')
    check(UID.fullmatch(record['transfer_syntax_uid']) and record['transfer_syntax_uid'] != '1.2.840.10008.1.2.1.99', 'transfer_syntax_invalid_or_deflated')
    px = record['pixel_data_header']
    check(set(px) == {'tag', 'vr', 'declared_length', 'value_offset'}, 'pixel_boundary_schema')
    check(px['tag'] in {'7fe00010', '7fe00008', '7fe00009'} and px['vr'] in {'OW', 'OB', 'OF', 'OD', None}, 'pixel_boundary_tag_vr')
    check(type(px['value_offset']) is int and 132 <= px['value_offset'] <= count, 'pixel_boundary_offset_out_of_prefix')
    check(type(px['declared_length']) is int and 0 <= px['declared_length'] <= 0xffffffff, 'pixel_boundary_length_invalid')
    if px['declared_length'] != 0xffffffff:
        check(px['value_offset'] + px['declared_length'] <= total, 'pixel_payload_exceeds_source_file')


def number_array(value, n):
    try:
        a = json.loads(value)
        if not isinstance(a, list) or len(a) != n or any(type(x) not in {int, float} or not math.isfinite(x) for x in a):
            return None
        return a
    except (ValueError, TypeError):
        return None


def plane(iop):
    a = number_array(iop, 6)
    if a is None:
        return {'plane': 'unavailable', 'reason': 'missing_or_invalid_six_vector'}
    row, col = a[:3], a[3:]
    rn = math.sqrt(sum(x*x for x in row)); cn = math.sqrt(sum(x*x for x in col)); dot = sum(x*y for x, y in zip(row, col))
    if abs(rn-1) > .002 or abs(cn-1) > .002 or abs(dot) > .002:
        return {'plane': 'invalid', 'reason': 'direction_vectors_not_orthonormal', 'row_norm': rn, 'column_norm': cn, 'dot': dot}
    normal = [row[1]*col[2]-row[2]*col[1], row[2]*col[0]-row[0]*col[2], row[0]*col[1]-row[1]*col[0]]
    strength = max(abs(x) for x in normal)
    label = ['sagittal', 'coronal', 'axial'][max(range(3), key=lambda x: abs(normal[x]))] if strength >= .90 else 'oblique'
    return {'plane': label, 'normal': normal, 'dominant_axis_strength': strength}


def description_plane(description):
    words = PLANE_WORD.findall(description)
    return sorted({('sagittal' if x.lower().startswith('sag') else 'coronal' if x.lower().startswith('cor') else 'axial') for x in words})


def audit_study(rows):
    uid = rows[0]['study_uid']; grouped = defaultdict(list)
    for r in rows:
        grouped[r['series_uid']].append(r)
    side_values = {k: sorted({r['header_fields'][k] for r in rows}) for k in SIDE_FIELDS}
    sides = sorted({v for values in side_values.values() for v in values if v})
    patients = sorted({r['patient_id_sha256'] for r in rows if r['patient_id_present']})
    descriptions = defaultdict(set)
    series_candidates = []
    legacy_mismatches = []
    n_desc_anomalies = 0
    plane_groups = Counter()
    for sid, group in sorted(grouped.items()):
        differences = {k: sorted({r['header_fields'][k] for r in group}) for k in SIDE_FIELDS if len({r['header_fields'][k] for r in group}) > 1}
        ids = sorted({r['patient_id_sha256'] for r in group if r['patient_id_present']})
        if len(ids) > 1:
            differences['patient_id_sha256'] = ids
        if len({r['patient_id_present'] for r in group}) > 1:
            differences['patient_id_blank_and_filled'] = True
        if differences:
            legacy_mismatches.append({'series_uid': sid, 'study_uid': uid, 'differences': differences})
        variants = {}
        for key in ['Rows', 'Columns', 'PixelSpacing', 'PatientPosition', 'SeriesDescription', 'SeriesNumber', 'AcquisitionNumber']:
            vals = sorted({r['header_fields'][key] for r in group})
            if len(vals) > 1:
                variants[key] = vals
        for r in group:
            if r['header_fields']['SeriesDescription']:
                descriptions[r['header_fields']['SeriesDescription']].add(sid)
        geometry = [{'sop_uid': r['sop_uid'], 'iop': r['header_fields']['ImageOrientationPatient'], **plane(r['header_fields']['ImageOrientationPatient'])} for r in group]
        gp = sorted({x['plane'] for x in geometry})
        plane_groups['/'.join(gp)] += 1
        declared = sorted({x for r in group for x in description_plane(r['header_fields']['SeriesDescription'])})
        mismatch = len(declared) == 1 and any(x in {'sagittal', 'coronal', 'axial'} and x != declared[0] for x in gp)
        if mismatch:
            n_desc_anomalies += 1
        frame_hashes = sorted({r['frame_of_reference_sha256'] for r in group if r['frame_of_reference_present']})
        if differences or variants or mismatch or len(gp) > 1 or any(x in {'invalid','unavailable'} for x in gp) or len(frame_hashes) > 1:
            series_candidates.append({'study_uid': uid, 'series_uid': sid, 'n_endpoints': len(group), 'side_patient_differences': differences, 'other_header_variants': variants, 'frame_hash_values': frame_hashes, 'frame_hash_varies': len(frame_hashes) > 1, 'description_values': sorted({r['header_fields']['SeriesDescription'] for r in group}), 'description_plane_tokens': declared, 'iop_geometry': geometry, 'description_iop_plane_mismatch': mismatch, 'candidate_only': True})
    repeated = {k: sorted(v) for k,v in descriptions.items() if len(v) > 1}
    mixed_id = len(patients) > 1
    blank_id = len({r['patient_id_present'] for r in rows}) > 1
    blank_sides = [k for k,v in side_values.items() if '' in v and len(v) > 1]
    noncanonical = [s for s in sides if s not in {'L','R'}]
    repeated_set = len(repeated) >= 3
    legacy_flag = len(sides) > 1 or mixed_id or blank_id or bool(blank_sides) or bool(noncanonical) or repeated_set
    legacy = {'study_uid': uid, 'nonempty_side_values': sides, 'mixed_nonempty_side_values': len(sides)>1, 'patient_id_sha256_values': patients, 'mixed_nonempty_patient_ids': mixed_id, 'patient_id_blank_and_filled': blank_id, 'laterality_blank_and_filled': '' in side_values['Laterality'] and len(side_values['Laterality'])>1, 'side_tag_values': side_values, 'side_tags_blank_and_filled': blank_sides, 'nonunilateral_or_unrecognized_side_values': noncanonical, 'repeated_acquisition_set_candidate': repeated_set, 'repeated_sequence_descriptions': repeated, 'candidate_only_requires_source_geometry_review': True}
    aliases = {'L':'L','LEFT':'L','R':'R','RIGHT':'R'}
    anatomical_aliases = sorted({aliases[s.upper()] for s in sides if s.upper() in aliases})
    flags = {'mixed_nonempty_patient_ids': mixed_id, 'patient_id_blank_and_filled': blank_id, 'explicit_left_and_right_aliases': anatomical_aliases == ['L','R'], 'side_tag_blank_and_filled': bool(blank_sides), 'noncanonical_side_values': bool(noncanonical), 'repeated_sequence_set_candidate': repeated_set, 'description_iop_anomaly': n_desc_anomalies>0, 'all_side_tags_blank': not sides, 'all_descriptions_blank': not descriptions}
    candidate = {**legacy, 'n_series':len(grouped), 'n_endpoints':len(rows), 'split': rows[0]['source_path'].split('/')[0], 'analytical_flags': flags, 'series_plane_group_counts':dict(plane_groups), 'n_description_iop_anomalies':n_desc_anomalies, 'side_alias_analysis_only':anatomical_aliases, 'priority_reason': 'patient_identity' if mixed_id or blank_id else 'explicit_both_sides' if flags['explicit_left_and_right_aliases'] else 'mixed_tags_and_repeated_sequences' if blank_sides and repeated_set else 'repeated_sequences' if repeated_set else 'description_iop' if n_desc_anomalies else 'tag_review'}
    return {'uid':uid, 'series_candidates':series_candidates, 'legacy_mismatches':legacy_mismatches, 'legacy_flag':legacy if legacy_flag else None, 'candidate':candidate if legacy_flag or n_desc_anomalies else None, 'patients':patients, 'flags':flags, 'n_series':len(grouped)}


def validate(repo, survey, out):
    check((survey/'summary.json').is_file() and (survey/'endpoint-headers.jsonl').is_file(), 'survey_final_artifacts_not_available')
    prep = json.loads((out/'inventory-validation.json').read_text())
    expected_path = out/'expected-endpoints.jsonl'
    check(file_sha(expected_path)==prep['expected_endpoints_sha256'], 'prepared_selection_file_changed')
    check(file_sha(survey/'manifest.json')==prep['manifest_sha256'], 'survey_manifest_changed_since_prepare')
    check(file_sha(Path(__file__))==prep['validator_sha256'], 'validator_changed_since_prepare')
    check(inventory_bindings(repo/'initiative/source/inventory')==prep['inventory_files'], 'inventory_changed_since_prepare')
    expected={r['path']:r for line in expected_path.open() if (r:=json.loads(line))}
    manifest=json.loads((survey/'manifest.json').read_text());summary=json.loads((survey/'summary.json').read_text());artifact=survey/'endpoint-headers.jsonl'
    check(summary['manifest_sha256']==file_sha(survey/'manifest.json') and summary['endpoint_artifact_sha256']==file_sha(artifact), 'summary_artifact_binding_mismatch')
    db=sqlite3.connect('file:'+str((survey/'checkpoint.sqlite3').resolve())+'?mode=ro',uri=True)
    db.execute('BEGIN')
    db_paths=set(); db_failures=[]
    for path,payload,sha in db.execute('SELECT path,payload,sha256 FROM records'):
        check(path in expected and digest(payload.encode())==sha, 'checkpoint_checksum_or_membership')
        check(json.loads(payload)['source_path']==path, 'checkpoint_payload_path_mismatch')
        db_paths.add(path)
    for path,code in db.execute('SELECT path,code FROM failures ORDER BY path'):
        check(path in expected and path not in db_paths and isinstance(code,str), 'checkpoint_failure_membership')
        db_failures.append({'source_path':path,'code':code})
    check(summary['failures']==db_failures, 'summary_failure_mismatch')
    seen=set(); previous_path=''; current=[]; current_key=None
    prefix_counts=Counter();syntax_counts=Counter(); blank_fields=Counter();header_counts={k:Counter() for k in ['Laterality','ImageLaterality','FrameLaterality','Manufacturer','ManufacturerModelName','PatientPosition']}
    n=0; full=0; prefix_bytes=0; series_total=0; study_total=0; missing_patient=0; missing_frame=0
    legacy_candidates=[];legacy_mismatches=[];studies=[];series_candidates=[];patient_studies=defaultdict(set); flags=Counter()
    def finish(group):
        nonlocal series_total,study_total
        if not group:return
        result=audit_study(group);series_total+=result['n_series'];study_total+=1
        legacy_mismatches.extend(result['legacy_mismatches'])
        if result['legacy_flag']:legacy_candidates.append(result['legacy_flag'])
        if result['candidate']:studies.append(result['candidate'])
        series_candidates.extend(result['series_candidates'])
        flags.update({k:int(v) for k,v in result['flags'].items()})
        for h in result['patients']:patient_studies[h].add((result['uid'],group[0]['source_path'].split('/')[0]))
    with artifact.open() as f:
        for line in f:
            check(bool(line.strip()),'blank_jsonl_line');record=json.loads(line);path=record['source_path']
            check(path in expected and path not in seen and path>previous_path,'output_membership_duplicate_or_sort')
            previous_path=path;seen.add(path);validate_record(record,expected[path])
            row=db.execute('SELECT payload,sha256 FROM records WHERE path=?',(path,)).fetchone()
            check(row is not None and json.loads(row[0])==record and digest(canonical(record))==row[1],'output_checkpoint_disagreement')
            key=(path.split('/')[0],record['study_uid'])
            if current_key is not None and key!=current_key:finish(current);current=[]
            current_key=key;current.append(record)
            n+=1;full+=record['full_file_sha256_verified'];prefix_counts[str(record['prefix_bytes'])]+=1;syntax_counts[record['transfer_syntax_uid']]+=1
            prefix_bytes+=sum(x['prefix_bytes'] for x in record['prefix_attempts'])
            missing_patient+=not record['patient_id_present'];missing_frame+=not record['frame_of_reference_present']
            for k,v in record['header_fields'].items():
                if not v:blank_fields[k]+=1
                if k in header_counts:header_counts[k][v]+=1
    finish(current);db.close()
    failure_paths={x['source_path'] for x in db_failures};check(seen==db_paths and seen|failure_paths==set(expected),'endpoint_success_failure_partition_incomplete')
    complete=seen==set(expected)
    check(summary['successful_endpoints']==n and summary['selected_endpoints']==len(expected) and summary['complete_endpoint_coverage']==complete,'summary_coverage_claim_mismatch')
    check(summary['selected_series']==24386 and summary['inventory_series']==24386 and summary['inventory_slices']==819635 and summary['source_revision']==REVISION,'summary_source_contract')
    check(summary['full_slice_header_coverage'] is False and summary['full_pixel_audit'] is False,'summary_scope_overclaim')
    check(summary['prefix_bytes_downloaded_for_successes']==prefix_bytes and summary['successful_prefix_size_counts']==dict(prefix_counts) and summary['endpoints_with_full_file_sha256_verified']==full,'summary_prefix_counts')
    check(sorted(summary['study_candidates'],key=lambda x:x['study_uid'])==sorted(legacy_candidates,key=lambda x:x['study_uid']),'summary_study_candidate_disagreement')
    check(sorted(summary['series_endpoint_mismatches'],key=lambda x:x['series_uid'])==sorted(legacy_mismatches,key=lambda x:x['series_uid']),'summary_series_mismatch_disagreement')
    cross=[{'patient_id_sha256':h,'n_studies':len(v),'studies':[{'study_uid':u,'split':s} for u,s in sorted(v)],'cross_train_test':len({s for _,s in v})>1,'candidate_only':True} for h,v in sorted(patient_studies.items()) if len(v)>1]
    for name,values in [('series-candidates.jsonl',series_candidates),('study-candidates.jsonl',studies)]:
        p=out/name;check(not p.exists(),'immutable_output_exists')
        with p.open('w') as f:
            for v in values:f.write(json.dumps(v,ensure_ascii=False,sort_keys=True)+'\n')
    dump(out/'cross-study-patient-id-groups.json',cross)
    result={'stage':'independent_final_endpoint_metadata_validation','agent_id':'/root/image_pipeline_audit','session_id':'/root/image_pipeline_audit','identity_kind':'codex_agent_task','workflow_run_id':'codex-20260920-rubric-v2','source_revision':REVISION,'input_bindings':[bind(p,repo.parent) for p in [survey/'manifest.json',survey/'summary.json',artifact,repo.parent/'header_endpoint_survey.py',Path(__file__),out/'inventory-validation.json',expected_path]],'integrity_validation_passed':True,'complete_endpoint_coverage':complete,'expected_endpoints':len(expected),'successful_endpoints':n,'failures':db_failures,'series_with_records':series_total,'studies_with_records':study_total,'checkpoint_payload_checksums_verified':len(db_paths),'exact_source_uid_size_oid_bindings_verified':n,'prefix_size_counts':dict(prefix_counts),'prefix_bytes_for_successes':prefix_bytes,'full_small_files_sha256_verified_as_recorded':full,'missing_patient_id_endpoints':missing_patient,'missing_frame_of_reference_endpoints':missing_frame,'cross_study_patient_hash_groups':len(cross),'cross_train_test_patient_hash_groups':sum(x['cross_train_test'] for x in cross),'original_summary_series_mismatches':len(legacy_mismatches),'original_summary_study_candidates':len(legacy_candidates),'independent_series_candidates':len(series_candidates),'independent_study_candidates':len(studies),'study_flag_counts':dict(flags),'description_iop_mismatch_series':sum(x['description_iop_plane_mismatch'] for x in series_candidates),'transfer_syntax_counts':dict(syntax_counts),'blank_header_field_endpoint_counts':dict(blank_fields),'header_value_counts':{k:dict(v) for k,v in header_counts.items()},'candidate_artifact_bindings':[bind(out/p,repo.parent) for p in ['series-candidates.jsonl','study-candidates.jsonl','cross-study-patient-id-groups.json']],'scope_limits':['No live job/code was modified; no new DICOM/network requests, raw pixels or diagnostic image reread.','Only lexical first/last SOP per series, not spatial endpoints or all-slice header coverage.','Independent file/checkpoint hashes establish artifact integrity and recorded source binding. Prefix bytes are not retained, so their recorded digests and parser/transport claims cannot be reverified from bytes here; full source suffixes remain unauthenticated by this endpoint audit.','PatientID hashes represent technical identifier equality, not verified real-world patient identity. Missing IDs never form patient-link groups.','Repeated descriptions or geometry, mixed side tags and description/IOP disagreement are candidates requiring source/anatomy review, not automatic side assignment, clinical labels or exclusions.','Description plane detection uses isolated explicit plane words; IOP class requires orthonormal vectors and dominant normal component>=0.90. Strongly oblique, missing and multi-plane descriptions remain separately limited.','Only whitelisted top-level fields preceding PixelData are in scope; enhanced nested per-frame and post-PixelData metadata are not assessed.'],'clinical_labels_accessed_or_changed':False}
    dump(out/'validation-summary.json',result)
    return {k:v for k,v in result.items() if k in ['integrity_validation_passed','complete_endpoint_coverage','successful_endpoints','series_with_records','studies_with_records','cross_study_patient_hash_groups','description_iop_mismatch_series','study_flag_counts']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--repo',type=Path,default=Path.cwd());parser.add_argument('--survey',type=Path);parser.add_argument('--output',type=Path)
    a=parser.parse_args();repo=a.repo.resolve();survey=(a.survey or repo.parent/'header-endpoint-full-v1').resolve();out=(a.output or Path(__file__).parent).resolve();out.mkdir(parents=True,exist_ok=True)
    result=prepare(repo,survey,out) if a.prepare else validate(repo,survey,out)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
