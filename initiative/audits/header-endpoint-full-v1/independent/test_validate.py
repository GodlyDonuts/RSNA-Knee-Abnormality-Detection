import copy
import hashlib
import unittest
from validate import validate_record, plane, description_plane, audit_study, FIELDS, EMPTY_SHA


def sample():
    expected={'path':'train_series/1.2.3/1.2.4/1.2.5.dcm','study_uid':'1.2.3','series_uid':'1.2.4','sop_uid':'1.2.5','sha256':'a'*64,'bytes':600000,'storage':'lfs'}
    fields=dict.fromkeys(FIELDS,'');fields.update(SeriesDescription='pd_tse_fs_cor_np',ImageOrientationPatient='[1,0,0,0,0,-1]',Rows='512',Columns='512')
    record={'source_path':expected['path'],'study_uid':'1.2.3','series_uid':'1.2.4','sop_uid':'1.2.5','source_sha256':expected['sha256'],'source_bytes':600000,'storage':'lfs','prefix_bytes':65536,'prefix_sha256':'b'*64,'full_file_sha256_verified':False,'verification_scope':'range_transport_and_uid_binding_only','prefix_attempts':[{'prefix_bytes':65536,'prefix_sha256':'b'*64}],'header_fields':fields,'patient_id_sha256':hashlib.sha256(b'one').hexdigest(),'patient_id_present':True,'frame_of_reference_sha256':hashlib.sha256(b'frame').hexdigest(),'frame_of_reference_present':True,'transfer_syntax_uid':'1.2.840.10008.1.2.1','pixel_data_header':{'tag':'7fe00010','vr':'OW','declared_length':524288,'value_offset':1500},'reached_pixel_data_header':True}
    return expected,record


class ValidatorTests(unittest.TestCase):
    def test_valid_endpoint_binding(self):
        e,r=sample();validate_record(r,e)
    def test_wrong_uid_rejected(self):
        e,r=sample();r['sop_uid']='1.2.99'
        with self.assertRaisesRegex(ValueError,'record_source_binding'):validate_record(r,e)
    def test_false_full_file_claim_rejected(self):
        e,r=sample();r['full_file_sha256_verified']=True
        with self.assertRaisesRegex(ValueError,'full_file_claim'):validate_record(r,e)
    def test_unreached_pixel_boundary_rejected(self):
        e,r=sample();r['reached_pixel_data_header']=False
        with self.assertRaisesRegex(ValueError,'pixel_header_not_proven'):validate_record(r,e)
    def test_pixel_value_outside_prefix_rejected(self):
        e,r=sample();r['pixel_data_header']['value_offset']=65537
        with self.assertRaisesRegex(ValueError,'pixel_boundary_offset'):validate_record(r,e)
    def test_declared_pixels_outside_file_rejected(self):
        e,r=sample();r['pixel_data_header']['declared_length']=600000
        with self.assertRaisesRegex(ValueError,'pixel_payload_exceeds'):validate_record(r,e)
    def test_prefix_history_tampering_rejected(self):
        e,r=sample();r['prefix_attempts'][0]['prefix_sha256']='c'*64
        with self.assertRaisesRegex(ValueError,'prefix_history_final_digest'):validate_record(r,e)
    def test_unwhitelisted_patient_value_rejected(self):
        e,r=sample();r['header_fields']['PatientID']='never-write'
        with self.assertRaisesRegex(ValueError,'header_whitelist'):validate_record(r,e)
    def test_missing_id_hash_consistency(self):
        e,r=sample();r['patient_id_present']=False
        with self.assertRaisesRegex(ValueError,'empty_hash_inconsistent'):validate_record(r,e)
        r['patient_id_sha256']=EMPTY_SHA;validate_record(r,e)
    def test_full_small_file_requires_full_digest(self):
        e,r=sample();e['bytes']=60000;r['source_bytes']=60000;r['prefix_bytes']=60000;r['prefix_attempts'][0]['prefix_bytes']=60000;r['full_file_sha256_verified']=True;r['verification_scope']='complete_file_sha256';r['pixel_data_header']['declared_length']=30000
        with self.assertRaisesRegex(ValueError,'full_file_hash_mismatch'):validate_record(r,e)
        r['prefix_sha256']=e['sha256'];r['prefix_attempts'][0]['prefix_sha256']=e['sha256'];validate_record(r,e)
    def test_three_orthogonal_planes(self):
        self.assertEqual(plane('[0,1,0,0,0,-1]')['plane'],'sagittal')
        self.assertEqual(plane('[1,0,0,0,0,-1]')['plane'],'coronal')
        self.assertEqual(plane('[1,0,0,0,1,0]')['plane'],'axial')
    def test_invalid_and_oblique_vectors_are_not_forced(self):
        self.assertEqual(plane('[2,0,0,0,1,0]')['plane'],'invalid')
        self.assertEqual(plane('')['plane'],'unavailable')
        self.assertEqual(plane('[0.70710678,-0.70710678,0,0,0,-1]')['plane'],'oblique')
    def test_description_token_boundaries(self):
        self.assertEqual(description_plane('pd_tse_fs_cor_np'),['coronal'])
        self.assertEqual(description_plane('AXIAL PD'),['axial'])
        self.assertEqual(description_plane('sag cor scout'),['coronal','sagittal'])
        self.assertEqual(description_plane('correction extra spectrum'),[])
    def test_description_iop_mismatch(self):
        _,r=sample();r['header_fields']['ImageOrientationPatient']='[0,1,0,0,0,-1]';a=audit_study([r])
        self.assertTrue(a['series_candidates'][0]['description_iop_plane_mismatch'])
    def test_mixed_patient_and_sides_are_candidates(self):
        _,r=sample();q=copy.deepcopy(r);q['sop_uid']='1.2.6';q['patient_id_sha256']=hashlib.sha256(b'two').hexdigest();q['header_fields']['Laterality']='R';r['header_fields']['Laterality']='L'
        a=audit_study([r,q]);self.assertTrue(a['flags']['mixed_nonempty_patient_ids']);self.assertTrue(a['flags']['explicit_left_and_right_aliases']);self.assertEqual(len(a['legacy_mismatches']),1)
    def test_missing_sides_do_not_imply_right(self):
        _,r=sample();a=audit_study([r]);self.assertTrue(a['flags']['all_side_tags_blank']);self.assertFalse(a['flags']['explicit_left_and_right_aliases']);self.assertIsNone(a['candidate'])
    def test_matching_multiple_contrasts_not_plane_anomaly(self):
        _,r=sample();q=copy.deepcopy(r);q['series_uid']='1.2.6';q['header_fields']['SeriesDescription']='t1_cor';a=audit_study([r,q]);self.assertEqual(a['series_candidates'],[])
    def test_repeated_three_descriptions_candidate_only(self):
        _,r=sample();rows=[]
        for i in range(6):
            q=copy.deepcopy(r);q['series_uid']=f'1.2.{i+10}';q['header_fields']['SeriesDescription']=['t1_cor','pd_cor','t2_cor'][i%3];rows.append(q)
        a=audit_study(rows);self.assertTrue(a['flags']['repeated_sequence_set_candidate']);self.assertFalse(a['flags']['explicit_left_and_right_aliases'])

if __name__=='__main__':unittest.main()
