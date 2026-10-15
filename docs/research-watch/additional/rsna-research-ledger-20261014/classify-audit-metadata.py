import json,re,collections
from pathlib import Path
D=Path('/workspace/shared/rsna-research-ledger-20261014')
a=json.loads((D/'audit-inventory.json').read_text())
F=[
('frozen_dinov3',r'^(dinov3-fold[12]|dinov3-threefold|epoch1-baseline|post-fold0-score)'),
('frozen_patch_attention',r'^dinov3-patch-attention'),
('rank32_fold0',r'^(dinov3-adapter-a32-(fold0|v1)|dinov3-large-adapter|adapter-update)'),
('rank32_all_development',r'^dinov3-adapter-a32-all-development'),
('rank_capacity',r'^(adapter-rank|dinov3-rank64)'),
('whole_series_and_context',r'^dinov3-whole-series'),
('slice_sampling',r'^(sampling-|stratified-slices)'),
('native384_all_development',r'^native384-all-development'),
('native384_fold_and_inputs',r'^(native|streaming-native384)'),
('effusion_supervision',r'^effusion-'),
('neighbor_channels',r'^neighbor-channels'),
('hard_patch_witness',r'^(patch-witness|witness-original|witness-initial)'),
('smooth_witness',r'^(smooth-witness|witness-FP16)'),
('plane_memory',r'^plane-memory'),
('cls_patch_mean',r'^cls-patch-summary'),
('public_labels_full',r'^public-label-(source|supervision-density|value-distance)'),
('public_labels_fixedmask',r'^public-label-fixedmask'),
('public_labels_coverage',r'^public-label-coverage'),
('public_labels_repeat',r'^public-label-repeat'),
('public_labels_all_development',r'^(public-label-final|final-public-source|final-fit-publisher|final-recovery)'),
('contrast',r'^(cache-contrast|robust-contrast|fitting-intensity)'),
('checkpoint_average',r'^epoch-weight-average'),
('source_slot',r'^source-slot'),
('window_bag',r'^source-window-bag'),
('patch_focus',r'^patch-focus'),
('dense_suffix',r'^dense-'),
('teacher_committee',r'^(public-teacher|committee-)'),
('slice_relations',r'^slice-relations'),
('anatomical_sharing',r'^anatomical-sharing'),
('patch_geometry',r'^patch-geometry'),
('ranking_support',r'^ranking-supervision'),
('complementary_selection',r'^fitting-series'),
('public_recipe_input_parity',r'^(public-dinov2|public-input|public-render|public-source|public-triplet)'),
('public_source_review',r'^(adapter-source|public-report-supervision|public-recipe|public-solution)'),
('mrnet_source_transfer',r'^(mrnet-|stanford-rsna|external-knee)'),
('mri_core',r'^mri-core'),
('kneecot',r'^kneecot'),
('larger_dinov3',r'^dinov3-larger'),
('early_backbone_screen',r'^competitive-pretrained'),
('early_pipeline_diagnostic',r'^(pilot-realdata|dinov3-h100|final-input-training)'),
('training_throughput',r'^throughput-profiler'),
('shared_control_equivalence',r'^(shared-control|current-control)'),
('run_cancellation',r'^training-preservation'),
]
clinical=r'^(cross-study-official-duplicate|bone-|cartilage|qualified|anatomical-plane|bilateral-|clinical-review|correspondence|conservative-semantic|daenerys-multimodal|daenerys-review|duplicate-consistency|endpoint-source|header-endpoint|independent-preread|independent-quality|last-twelve|laterality|mixed-laterality|patient-|post-review|preread-|primary74|quality-native|readiness-|repeated-|report-|review-|reviewed-series|semantic-|series-equivalence|series-preference|source-completeness|supervision-|targeted-image|technical-|tier4|visual-|literal-test|final-(direct|release|persisted|runtime|semantic|supervision)|release-admission|weak-split|legacy-label|image-review)'
infra=r'^(cache-(progress|resume|v3)|checkpoint-|code-validation|competitive-(inference|ready|runtime)|dinov3-(capacity|delta|deployment|large-access|official)|evc|full-cache|full-image|github-source|h100-|newton-|obsolete-|process-memory|published-package|reconnected-training|research-|second-node|source-(pixel|preflight|vs)|storage-copy|pilot-plane|slice-continuity)'
rows=[]
for e in a['entries']:
 p=e['path'];fam=next((name for name,pat in F if re.search(pat,p)),None)
 if fam: cat='model_or_input_family'
 elif re.search(clinical,p):cat='annotation_or_data_governance_excluded'
 elif p.startswith(('hf-','kaggle-','daily-kaggle-','final-publication-','dinov3-adapter-a32-kaggle-')):cat='deployment_or_preservation_receipt'
 elif re.search(infra,p) or p=='competitive-modeling-research-v2':cat='general_engineering_or_planning'
 else:cat='remaining_metadata_triage'
 rows.append(dict(path=p,type=e['type'],sha=e['sha'],category=cat,family=fam or '',content_review_claim='metadata_only_unless_listed_in_review_register'))
print(collections.Counter(r['category'] for r in rows))
print('REMAINING:',[r['path'] for r in rows if r['category']=='remaining_metadata_triage'])
for fam,pat in F:
 rr=[r for r in rows if r['family']==fam]
 print(fam,len(rr),','.join(r['path'] for r in rr))
(D/'supplement-metadata-classification.json').write_text(json.dumps({'snapshot':a['snapshot'],'classification_is_name_based_not_content_review':True,'entries':rows},indent=2)+'\n')
