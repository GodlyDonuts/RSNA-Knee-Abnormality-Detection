"""Source-bind unchanged candidate reviews; labels and clinical truth stay separate."""
import collections,hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent;packet=json.loads((D/'packet.json').read_text());images=packet['images'];assert len(images)==62
hosted=[json.loads((D/f'upload-{i:02d}.result.json').read_text())['structuredContent'] for i in range(1,63)];index={r['reference']:(r,s) for r,s in zip(hosted,images,strict=True)}
manifest_sha=hashlib.sha256((D/'image-only-manifest.json').read_bytes()).hexdigest();rows=[];originals=[]
from rsna.schema import TARGETS
for name,size,sha in [('provisional-001.original.json',209643,'d454a3fdfc72fbe82608f22678567a458f53ede5b899121320e1d68321cfca01'),('report-informed-002.original.json',217870,'4a1047e391349708f2ec184587156fd466ecf2521fafd4bc2fa310feaebe37c3')]:
 raw=(D/name).read_bytes();assert len(raw)==size and hashlib.sha256(raw).hexdigest()==sha
 v=json.loads(raw);originals.append(v);assert v['packet_id']==packet['packet_id'] and v['source']['study_uid']==packet['study_uid'] and v['source']['source_revision']==packet['source_revision'] and v['source']['image_manifest']['sha256']==manifest_sha and v['candidate_annotation_only'] is True and v['accepted_labels_changed'] is False
 c=v['coverage'];assert c['expected_references']==c['personally_inspected_references']==62 and c['unique_source_slices']==342 and c['inaccessible_references']==[] and len(c['access_log'])==62 and len({a['reference'] for a in c['access_log']})==62
 for a in c['access_log']:
  r,s=index[a['reference']];assert a['sha256']==r['sha256']==s['sha256'] and a['byte_size']==r['byte_size']==s['size'] and (a['series_index'],a['slice_start'],a['slice_end'])==(s['series_index'],s['slice_start'],s['slice_end'])
  series=packet['series'][s['series_index']-1];assert a['original_sop_uids']==series['ordered_sop_uids'][s['slice_start']-1:s['slice_end']] and a['original_source_sha256']==series['source_sha256'][s['slice_start']-1:s['slice_end']]
 assert [t['target'] for t in v['targets']]==list(TARGETS);quotes=evidence=0
 for t in v['targets']:
  assert t['status'] in ('positive','negative','uncertain','not_assessable') and t['confidence'] in ('low','medium','high') and t['confidence_is_calibrated'] is False
  for q in t['literal_report_quotes']:assert q in packet['report'];quotes+=1
  for e in t['supporting_image_evidence']+t['contrary_image_evidence']:
   si=e['series'];assert type(si) is int and 1<=si<=10 and all(type(n) is int and 1<=n<=packet['series'][si-1]['slice_count'] for n in e['slices']);evidence+=1
 rows.append({'file':name,'bytes':size,'sha256':sha,'coverage_records':62,'targets':12,'literal_report_quotes_matched':quotes,'valid_series_slice_evidence_records':evidence,'states':dict(collections.Counter(t['status'] for t in v['targets']))})
a,b=originals;assert a['reading_order']['report_opened'] is False and a['reading_order']['early_report_access'] is False and not any(t['literal_report_quotes'] for t in a['targets'])
assert b['preserved_provisional']['sha256']==rows[0]['sha256'] and b['preserved_provisional']['bytes']==rows[0]['bytes'] and b['preserved_provisional']['original_bytes_unchanged'] is True
assert b['report']['text']==packet['report'] and b['report']['sha256']==packet['report_sha256']
changes=[x['target'] for x,y in zip(a['targets'],b['targets'],strict=True) if x['status']!=y['status']];assert changes==['Medial Meniscus']
out={'status':'exact_original_candidate_reviews_source_and_complete_coverage_bound','originals':rows,'status_changes_after_report':changes,'all10series_342slices_62refs_bound':True,'image_before_report_order_reviewer_reported':True,'independent_underlying_tool_call_trace_collected':False,'not_clinical_gold_or_model_accuracy':True,'accepted_labels_changed':False,'additional_original_views_requested':14}
with (D/'original-reviews-validated.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
