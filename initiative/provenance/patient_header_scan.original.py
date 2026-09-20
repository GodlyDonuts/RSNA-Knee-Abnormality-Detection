from pathlib import Path
import os,json,io,hashlib,time,re,sys
import pandas as pd,pydicom
from dotenv import load_dotenv
from rsna.preprocess import Source
base=Path(__file__).resolve().parent; repo=base/'recovered'
load_dotenv(base.parent/'.env')
inv=pd.read_parquet(repo/'initiative/source/inventory')
selected=inv.groupby(['split','study_uid'],sort=True).head(1).sort_values(['split','study_uid']).copy()
del inv
out=base/'patient_header_scan.jsonl';seen=set()
if out.exists():
 for line in out.read_text().splitlines(): seen.add(json.loads(line)['study_uid'])
remaining=selected[~selected.study_uid.isin(seen)]
src=Source(None,remaining.set_index('path')[['sha256','bytes','storage']].to_dict('index'),revision='52f995f4568e36bfa5fbbaf69a89af41226b4f5d')
def digest(value):return hashlib.sha256(value.encode()).hexdigest() if value else None
start=time.monotonic()
with out.open('a',buffering=1) as log:
 for offset in range(0,len(remaining),32):
  block=remaining.iloc[offset:offset+32]
  for row,data in zip(block.itertuples(),src.read_series(block.path.tolist())):
   d=pydicom.dcmread(io.BytesIO(data),stop_before_pixels=True)
   if (str(d.StudyInstanceUID),str(d.SeriesInstanceUID),str(d.SOPInstanceUID))!=(row.study_uid,row.series_uid,row.sop_uid):raise ValueError('Header UID mismatch')
   patient=str(d.get('PatientID','')).strip();issuer=str(d.get('IssuerOfPatientID','')).strip()
   record={'study_uid':row.study_uid,'series_uid':row.series_uid,'sop_uid':row.sop_uid,'split':row.split,'source_sha256':row.sha256,'patient_id_sha256':digest(patient),'issuer_sha256':digest(issuer),'patient_namespace_sha256':digest(issuer+'|'+patient) if patient else None,'patient_id_length':len(patient),'patient_id_numeric':patient.isnumeric(),'patient_id_equals_study_uid':patient==row.study_uid,'laterality':str(d.get('Laterality','')),'image_laterality':str(d.get('ImageLaterality','')),'manufacturer':str(d.get('Manufacturer','')),'model':str(d.get('ManufacturerModelName',''))}
   log.write(json.dumps(record)+'\n')
  print(f'patient headers={len(seen)+min(offset+32,len(remaining))}/{len(selected)} elapsed={time.monotonic()-start:.1f}s',flush=True)
src.downloads.shutdown()
frame=pd.read_json(out,lines=True);dest=repo/'initiative/audits/patient-header-audit';dest.mkdir(exist_ok=True)
frame.to_parquet(dest/'study_header_index.parquet',index=False)
groups=frame[frame.patient_namespace_sha256.notna()].groupby('patient_namespace_sha256').filter(lambda f:len(f)>1)
groups.to_parquet(dest/'candidate_patient_groups.parquet',index=False)
report={'scope':'One verified DICOM header per released study; pseudonymous identifiers do not prove true patient identity or within-study consistency','source_revision':'52f995f4568e36bfa5fbbaf69a89af41226b4f5d','studies_checked':len(frame),'expected_studies':len(selected),'complete_study_coverage':set(frame.study_uid)==set(selected.study_uid),'missing_patient_ids':int(frame.patient_id_sha256.isna().sum()),'missing_issuers':int(frame.issuer_sha256.isna().sum()),'patient_identifiers_equal_study_uid':int(frame.patient_id_equals_study_uid.sum()),'repeated_identifier_groups':int(groups.patient_namespace_sha256.nunique()),'studies_in_repeated_identifier_groups':len(groups),'cross_train_test_identifier_groups':int(groups.groupby('patient_namespace_sha256')['split'].nunique().gt(1).sum()),'patient_id_length_distribution':{str(k):int(v) for k,v in frame.patient_id_length.value_counts().items()},'study_header_index_sha256':hashlib.sha256((dest/'study_header_index.parquet').read_bytes()).hexdigest()}
(dest/'patient_header_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
