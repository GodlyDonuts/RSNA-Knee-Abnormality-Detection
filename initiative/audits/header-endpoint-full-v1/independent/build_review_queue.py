"""Metadata-only UID-to-blinded-batch mapping; no report inference or target values."""
from pathlib import Path
from collections import Counter
import json
from validate import bind, dump, file_sha
repo=Path.cwd();base=repo.parent;out=Path(__file__).resolve().parent
candidates=[json.loads(x) for x in (out/'study-candidates.jsonl').read_text().splitlines()]
series=[json.loads(x) for x in (out/'series-candidates.jsonl').read_text().splitlines()]
needed={x['study_uid'] for x in candidates};locations={};inputs=[]
for path in sorted((repo/'initiative/reviews/inputs').glob('batch-*.json')):
    for index,row in enumerate(json.loads(path.read_text())):
        uid=row['study_uid']
        if uid in needed:
            assert uid not in locations
            locations[uid]={'source_batch_path':str(path.relative_to(repo)),'source_batch_index':index,'source_batch_sha256':file_sha(path)}
    inputs.append({'path':str(path.relative_to(repo)), 'sha256':file_sha(path)})
queue=[]
for item in candidates:
    flags=item['analytical_flags'];reason=item['priority_reason']
    priority=1 if flags['mixed_nonempty_patient_ids'] or flags['patient_id_blank_and_filled'] else 2 if flags['explicit_left_and_right_aliases'] or flags['repeated_sequence_set_candidate'] else 3 if flags['description_iop_anomaly'] else 4
    queue.append({'study_uid':item['study_uid'],**locations.get(item['study_uid'],{}),'priority_tier':priority,'priority_reason':reason,'n_series':item['n_series'],'raw_side_values':item['nonempty_side_values'],'blank_and_filled_side_fields':item['side_tags_blank_and_filled'],'flags':flags,'description_iop_series_uids':[x['series_uid'] for x in series if x['study_uid']==item['study_uid'] and x['description_iop_plane_mismatch']],'required_next_step':'Read complete original source for correspondence; inspect all-series endpoint metadata; resolve with bounded full native anatomy only when candidate warrants it. Preserve independent source/visual assessments. No automatic label or side changes.'})
queue.sort(key=lambda x:(x['priority_tier'],x['study_uid']))
a={'stage':'metadata_candidate_review_queue','agent_id':'/root/image_pipeline_audit','session_id':'/root/image_pipeline_audit','validation_summary_binding':bind(out/'validation-summary.json',base),'study_candidate_binding':bind(out/'study-candidates.jsonl',base),'series_candidate_binding':bind(out/'series-candidates.jsonl',base),'source_batch_bindings':inputs,'n_candidate_studies':len(queue),'priority_tier_counts':dict(Counter(x['priority_tier'] for x in queue)),'queue':queue,'scope':'Only study UID fields were used to map existing technical candidates to blinded batch locations. No report-based target inference, annotations or official labels were opened or changed. Priority is for technical review, not established bilateral anatomy or exclusion.'}
dump(out/'candidate-review-queue.json',a)
print(json.dumps({'n_candidate_studies':len(queue),'priority_tier_counts':a['priority_tier_counts'],'unmapped_studies':[x['study_uid'] for x in queue if 'source_batch_path' not in x]},indent=2))
print('DESCRIPTION_IOP_BATCH_LOCATIONS')
for x in queue:
    if x['flags']['description_iop_anomaly']:print(json.dumps({k:x[k] for k in ['study_uid','source_batch_path','source_batch_index','description_iop_series_uids'] if k in x}))
