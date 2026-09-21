"""Compare final endpoint metadata with previously validated bounded native studies."""
from pathlib import Path
import json
from validate import file_sha, dump, bind, check

repo=Path.cwd();base=repo.parent;out=Path(__file__).resolve().parent;artifact=base/'header-endpoint-full-v1/endpoint-headers.jsonl'
check(artifact.exists(),'final_endpoint_artifact_not_available')
prep=json.loads((out/'inventory-validation.json').read_text())
expected={r['path']:r for line in (out/'expected-endpoints.jsonl').open() if (r:=json.loads(line))}
references=[];lookup={}
for name in ['bilateral-067037','bilateral-018019','bilateral-017007']:
    p=base/name/'all-slice-headers-and-pixels.jsonl';rows=[json.loads(x) for x in p.read_text().splitlines()];uids={r['study_uid'] for r in rows};check(len(uids)==1,'reference_multiple_studies');study=next(iter(uids))
    selected={path:e for path,e in expected.items() if e['study_uid']==study};rr={r['source_path']:r for r in rows};check(set(selected)<=set(rr),'reference_missing_selected_sop')
    references.append({'reference_binding':bind(p,base),'study_uid':study,'n_full_reference_slices':len(rows),'n_selected_endpoints':len(selected)})
    for path,e in selected.items():
        check(path not in lookup,'reference_path_overlap');lookup[path]=(rr[path],e,name)
compared=[];mismatches=[];prefix_compared=0
for line in artifact.open():
    r=json.loads(line);path=r['source_path']
    if path not in lookup:continue
    old,e,name=lookup[path];bad=[]
    for key in ['study_uid','series_uid','sop_uid','source_sha256','header_fields','patient_id_sha256','frame_of_reference_sha256']:
        if r[key]!=old[key]:bad.append(key)
    if r['source_bytes']!=e['bytes']:bad.append('source_bytes_vs_inventory')
    prefix_verified=False
    if 'source_prefix_65536_sha256' in old:
        candidate=next((x['prefix_sha256'] for x in r['prefix_attempts'] if x['prefix_bytes']==65536),None)
        if candidate!=old['source_prefix_65536_sha256']:bad.append('independently_full_download_recomputed_65536_prefix')
        prefix_verified=True;prefix_compared+=1
    compared.append({'source_path':path,'reference_name':name,'exact_metadata_and_source_binding_match':not bad,'prefix_hash_independently_comparable':prefix_verified})
    mismatches.extend({'source_path':path,'field':x} for x in bad)
check({x['source_path'] for x in compared}==set(lookup),'final_artifact_reference_coverage_missing')
result={'stage':'independent_endpoint_comparison_to_preserved_full_native_references','agent_id':'/root/image_pipeline_audit','session_id':'/root/image_pipeline_audit','endpoint_artifact_binding':bind(artifact,base),'comparison_script_binding':bind(Path(__file__).resolve(),base),'references':references,'n_selected_endpoints':len(compared),'n_exact_metadata_source_matches':sum(x['exact_metadata_and_source_binding_match'] for x in compared),'n_prefix_hashes_compared_to_independently_downloaded_full_files':prefix_compared,'mismatches':mismatches,'comparisons':compared,'limits':'Full native reference artifacts were previously produced during bounded studies; this is a metadata correspondence recheck, not a new image reread or full-corpus pixel audit. Only017007 retained a digest of the exact65536-byte prefix recomputed during full-file validation; other reference full-file/pixel digests are not compared to prefix digests. No fresh network downloads or clinical labels.'}
dump(out/'full-reference-comparison.json',result)
print(json.dumps({k:result[k] for k in ['n_selected_endpoints','n_exact_metadata_source_matches','n_prefix_hashes_compared_to_independently_downloaded_full_files','mismatches']},indent=2))
