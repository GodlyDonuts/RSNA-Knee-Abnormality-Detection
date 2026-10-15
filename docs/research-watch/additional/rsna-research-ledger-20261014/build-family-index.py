import json,collections,csv
from pathlib import Path
D=Path('/workspace/shared/rsna-research-ledger-20261014')
a=json.loads((D/'supplement-metadata-classification.json').read_text()); entries=a['entries']; pin=a['snapshot']; base='https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/'+pin+'/'
# Each row summarizes a family, not each receipt in the corresponding directory group.
F={
'frozen_dinov3':('Frozen DINOv3 CLS','Completed three folds; internal pair-weighted macro 0.7522657162. Five additional fold2 public scores are now explicitly listed in the supplement.','initiative/audits/kaggle-checkpoint-comparison-v1/README.md'),
'frozen_patch_attention':('Frozen patch-attention replacement','Completed three folds; aggregate macro 0.7360400218. Fold0 and fold2 declined; no wholesale replacement.','initiative/audits/dinov3-patch-attention-fold2-result-v1/README.md'),
'rank32_fold0':('Rank32 encoder adaptation','Completed fixed ten epochs; internal 0.8620940794 and public 0.769. Projection-update diagnostics support real adaptation, not isolated LoRA causality.','initiative/audits/dinov3-adapter-a32-fold0-result-v1/README.md'),
'rank32_all_development':('Rank32 224 all-development fit','Epoch10 / 4260 updates privately verified; public 0.792. No internal holdout. Explicit completion added.','initiative/audits/dinov3-adapter-a32-all-development-v1/epoch-publication/epoch-010/snapshot.json'),
'rank_capacity':('Rank64 and stricter rank proposal','Original rank64 completed; internal 0.8340496537, public 0.778. Stricter common-initialization paired-seed rank experiment remains proposed.','initiative/audits/adapter-rank-learning-v1/README.md'),
'whole_series_and_context':('Whole-series coverage and adjacent-feature context','No-context all-series/all-slice control completed at 0.8603399208. The distinct 66433-parameter adjacent-feature context production contrast remains prepared only.','initiative/experiments/competitive-v1/whole-series-context-v1/README.md'),
'slice_sampling':('Uniform versus stratified slices','Both completed; internal delta +0.0009106945 inconclusive; public 0.770 versus 0.784.','initiative/audits/sampling-fixed-final-v1/README.md'),
'native384_all_development':('Original-label native384 all-development fit','Job868093 completed0:0, ten epochs / 4260 updates; public E5 0.804 and E10 0.802. No internal holdout.','initiative/audits/native384-all-development-epochs-v1/README.md'),
'native384_fold_and_inputs':('Native384 fold0 and native-cache work','Completed fold0 at 0.8630558226 versus uniform224 0.8606037925; inconclusive. Cache, reconstruction and streaming receipts are supporting engineering, not additional independent fits.','initiative/audits/native384-fixed-final-v1/README.md'),
'effusion_supervision':('Reviewed Effusion supervision across two seeds','Four full three-epoch arms completed. Neither seed established primary Effusion gain; seed2 macro declined. Early initialization and three deployment failures remain distinct.','initiative/audits/effusion-supervision-seed2-final-v1/README.md'),
'neighbor_channels':('Gray versus physical and distant neighboring channels','Three full three-epoch arms completed; primary MM contrasts inconclusive. Original gray remains the default.','initiative/audits/neighbor-channels-full-fit-v1/README.md'),
'hard_patch_witness':('Original hard top-k signed patch witness','Real recovery engineering is reported in subsequent summaries; no completed full accuracy contrast identified for this original BF16 hard selector. Initial tied-score gradient coverage 0.0868056%.','initiative/audits/witness-initial-gradient-v1/README.md'),
'smooth_witness':('Smooth local witness','Completed three-epoch control/candidate; primary MM delta +0.0008348678 inconclusive. FP16/scaler and hard-selector recovery are not separate accuracy wins.','initiative/audits/smooth-witness-full-fit-v2/README.md'),
'plane_memory':('Plane-memory residual','Completed control/candidate; macro delta +0.0042802833 inconclusive. Secondary MM gain cannot replace the primary endpoint.','initiative/audits/plane-memory-full-fit-v1/README.md'),
'cls_patch_mean':('CLS plus mean-patch residual','Completed ten-epoch control/candidate; macro fell about 0.0186373. Reject this tested mean branch.','initiative/audits/cls-patch-summary-full-fit-v2/README.md'),
'public_labels_full':('Full public-label supervision','Completed control/candidate; macro 0.8997840648 versus 0.8786705965, delta +0.0211134683. Retain known Steven-v6/Gold58-guided source ancestry and sparse-support caveats.','initiative/audits/public-label-source-full-fit-preparation-v1/paired-fixed-final/macro-support-diagnostic.json'),
'public_labels_fixedmask':('Public values on original masks','Completed candidate; macro 0.8536812809, negative versus shared control. Not an image-teacher blending experiment.','initiative/audits/public-label-fixedmask-production-20261010-v1/README.md'),
'public_labels_coverage':('Coverage-only public supervision','Completed candidate; macro 0.8693023133, primary difference inconclusive and MM lower. Not equivalent to full public supervision.','initiative/audits/public-label-coverage-production-20261011-v1/README.md'),
'public_labels_repeat':('Public-label second-seed repeat','Control endpoint 0.8691356747; public arm finished2810 training updates but fixed-final evaluation remains unobserved after parent terminal. No replication claim.','docs/current-state.md'),
'public_labels_all_development':('All-development native-public fit','Original908454 is partial at the snapshot; eight epochs preserved and epoch9 underway. Public E1/E3/E4/E5/E6 are 0.752/0.856/0.865/0.872/0.872.','docs/current-state.md'),
'contrast':('Robust cache contrast','Completed ten-epoch isolated p1–p99 contrast at 0.8537006030 versus 0.8786705965. Input-parity and full-cohort coverage checks do not alter this negative outcome.','initiative/audits/cache-contrast-production-20261008-v1/README.md'),
'checkpoint_average':('Equal E6–E10 weight average','Completed evaluation without new fitting; macro 0.8990962409, no established gain over native-public E10.','initiative/audits/epoch-weight-average-fixed-final-20261014-v1/README.md'),
'source_slot':('Older-source-slot DINOv3 recipe','Completed ten epochs; macro 0.8782870962, below contextual native-public 0.8997840648. Many recipe elements differ.','initiative/audits/source-slot-fixed-final-20261014-v1/README.md'),
'window_bag':('Four-group mean versus query window bag','Completed actual GPU recovery engineering; no completed full-fit accuracy contrast identified. New joint-group training control is distinct from source-slot.','initiative/audits/source-window-bag-GPU-terminal-20261014-v1/README.md'),
'patch_focus':('Disease-query residual patch focus','Actual CPU MRI/model admission completed; GPU917001 pending at pinned snapshot and target relocation incomplete. No completed training result.','docs/current-state.md'),
'dense_suffix':('Dense suffix versus suffix rank32','Real production started; dense canceled by parent completion after last905 updates/three epochs. Not planned exit75 or a completed negative contrast. Matched suffix full endpoint unestablished.','initiative/audits/training-preservation-cancellation-20261014-v1/README.md'),
'teacher_committee':('Public image-teacher and twenty-member committee','Eight-study CPU/GPU evidence and bit-exact twenty-member shared-prefix forward accepted. Full collector terminal and student fitting remain unestablished.','initiative/audits/public-teacher-shared-prefix-GPU-20261013-v1/README.md'),
'slice_relations':('Physical slice-distance auxiliary objective','Full-cohort geometry and H100 initialization-gradient scale measured. Failed original compiler probe followed by corrected completed probe. No accepted full production accuracy contrast.','initiative/audits/slice-relations-gradient-header-recovery-v1/training-only-scale-decision.json'),
'anatomical_sharing':('Logical anatomical task sharing','Fitting-only aggregate support measured; shared-query and disease-residual mechanism not integrated or trained. Do not count logical unions as new rare-disease labels.','initiative/audits/anatomical-sharing-support-v1/README.md'),
'patch_geometry':('Patch geometry and padding hypothesis','All12476 fitting-series geometry audited; mean wholly artificial patch fraction0.9248%. Negative diagnostic for letterboxing as a main explanation; no registration or accuracy claim.','initiative/audits/patch-geometry-research-v1/README.md'),
'ranking_support':('Ranking-objective pair support','Measured one-study forwards yield zero same-target pairs; effective-eight support sparse for rare targets. No ranking-loss fit completed.','initiative/audits/ranking-supervision-exposure-v1/README.md'),
'complementary_selection':('Complementary series selection','214/2247 fitting studies omit an available sagittal/coronal non-fluid category. Same-budget revised selector proposed; no isolated trained contrast found.','initiative/audits/fitting-series-coverage-20261007-v1/README.md'),
'public_recipe_input_parity':('Source-faithful public rendering and triplets','Completed eight-study source/input comparisons and exact triplet adapter checks; later source-slot supplies the trained bundled recipe result. Individual parity/retrieval fixes are engineering.','initiative/audits/public-dinov2-real-inputs-20261013-v2/README.md'),
'public_source_review':('Public source provenance and recipe review','Aggregate source collection and identity review only. Active Shio bytes are Steven-v6 report-label blend; no documented MRI-teacher ancestor.','initiative/audits/adapter-source-research-v1/sources.json'),
'mrnet_source_transfer':('MRNet source fit and possible transfer','Ten source epochs completed: ACL0.8621082621 / meniscus0.7215843858 / abnormality0.9285087719. RSNA transfer not completed; overlap and rights gates distinct.','initiative/audits/mrnet-full-source-final-v1/README.md'),
'mri_core':('MRI-CORE prototype','Scaffolding/intake only; no accepted pretrained parity or knee classification result.','initiative/audits/mri-core-prototype-intake-20261012-v1/README.md'),
'kneecot':('KneeCoT private intake','Bounded intake engineering only; no training or RSNA improvement. Clinical records deliberately not inspected here.','initiative/audits/kneecot-private-research-access-20261006-v1/README.md'),
'larger_dinov3':('Larger DINOv3 backbones','Config/meta feasibility only; no H+ or7B weight acquisition/training result established.','initiative/audits/dinov3-larger-backbone-feasibility-v1/README.md'),
'early_backbone_screen':('Early ConvNeXt and DINOv2 screen','ConvNeXt actual pretrained CPU forward/head backward on synthetic input; DINOv2 header topology only. No current clinical comparison.','initiative/audits/competitive-pretrained-local-check-v1/convnext-actual-cpu-receipt.json'),
'early_pipeline_diagnostic':('Early CNN pipeline and first DINOv3 pilot','126-update pipeline smoke and26-update two-H100 DINOv3 pilot completed recovery checks. They are diagnostic, not competitive fits. Protected acceptance records were not read.','initiative/audits/dinov3-h100-pilot-v6/README.md'),
'training_throughput':('Training throughput profiler','Prepared/corrected implementation and64 root tests; no actual isolated MRI GPU setting promoted. Measured existing-run timings are motivation, not new profiler benchmarks.','initiative/experiments/competitive-v1/throughput-profile-v1/README.md'),
'shared_control_equivalence':('Exact common-control equivalence','Final learned tensors, optimizer/scaler/RNG/history and final prediction bytes were equal across two completed controls. Reuse requires recipe compatibility; this is not a new accuracy trial.','initiative/audits/shared-control-final-20261008-v1/README.md'),
'run_cancellation':('Training cancellation and preservation','Parent899766 terminal canceled dense overlap by signal15/143; repeat final evaluation unobserved. Already-preserved source-slot result stays complete.','initiative/audits/training-preservation-cancellation-20261014-v1/README.md'),
}
assert set(F)=={r['family'] for r in entries if r['family']}, (set(F)^ {r['family'] for r in entries if r['family']})
lines=['# RSNA experiment family coverage','',f'Pinned repository snapshot: {pin}.','',
'This index reconciles the meaningful model, input and supervision family names in the complete nonrecursive audit inventory with the existing ledger and the historical expansion. A family is a navigation group, not a count of independent experiments. A matching folder name is metadata evidence only; lifecycle claims below come from the cited aggregate summaries or the already-reconciled ledger. No claim is made that every member receipt was read.','',
'## Metadata scope','',
'All 1,090 root entries are assigned a navigation category: 230 entries in 43 model/input/evidence groups; 313 deployment or preservation entries; 484 annotation or data-governance entries; and 63 general engineering or planning entries. These categories are a new deterministic name-based taxonomy, not a correction to the earlier index’s different grouping. Counts are not training-run counts. Annotation/case-level contents were excluded. Reviewed submission outcomes are explicitly reconciled even though their directories are in the deployment category.','',
'## Family reconciliation','',
'| Family | Matching audit entries | Pinned lifecycle and ledger interpretation | Evidence |','|---|---:|---|---|']
for key,(title,status,src) in F.items():
 n=sum(r['family']==key for r in entries)
 lines.append(f'| {title} | {n} | {status} | [source]({base}{src}) |')
lines += ['', '## Experiment protocol directory reconciliation','',
'The complete pinned competitive-v1 tree contains 22 directories. All are accounted for below; the folds directory is metadata, leaving 21 protocol/diagnostic directories. Several directories belong to one family, and some completed families live only under audits.','',
'| Directory | Mapping |','|---|---|']
P={
'adapter-rank-capacity-v1':'Completed original rank64 comparison; stricter causal repeat remains proposed',
'cache-contrast-v1':'Completed negative isolated contrast',
'cls-patch-summary-v1':'Completed negative mean-patch residual',
'dense-suffix-capacity-v1':'Actual dense production interrupted; comparison incomplete',
'effusion-review-contrast-v1':'Four completed arms across two seeds; no replicated macro gain',
'epoch-weight-average-v1':'Completed evaluation; no new fit',
'folds':'Population/split metadata; contents excluded from this expansion',
'native-detail-v1':'Native-cache engineering followed by completed fold0 and all-development fits',
'native-resolution-profile-v1':'Metadata-only motivation for native detail; not another fit',
'neighbor-channels-v1':'Three completed arms; no established primary gain',
'patch-attention-v1':'Three completed frozen patch-attention folds; negative aggregate',
'patch-witness-v1':'Original hard-selector engineering; smooth successor has separate completed comparison',
'public-input-recipe-v1':'Rendering/input engineering; trained bundled recipe is source-slot',
'public-label-final-fit-v1':'All3403 native-public run incomplete at snapshot; five checkpoints scored',
'public-label-source-v1':'First control/candidate completed; second-seed evaluation incomplete',
'slice-relations-v1':'Geometry/gradient diagnostics; no completed full accuracy fit',
'source-matched-slot-v1':'Completed ten-epoch result0.8782870962',
'source-patch-focus-v1':'CPU accepted; GPU/production not completed at snapshot',
'source-window-bag-v1':'Recovery accepted; no full-fit result',
'stratified-slices-v1':'Uniform/stratified completed',
'throughput-profile-v1':'Prepared tested profiler; no actual promoted GPU setting',
'whole-series-context-v1':'No-context coverage control completed; adjacent-context production remains proposed',
}
for name,status in P.items():lines.append(f'| [{name}](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/tree/{pin}/initiative/experiments/competitive-v1/{name}) | {status} |')
lines += ['', '## Finite remaining receipt checks','',
'1. Public-label repeat: locate an original source-bound fixed-final candidate evaluation or an explicit evaluation failure. The preserved epoch10 checkpoint alone is insufficient.',
'2. Dense/suffix comparison: reconcile the last promoted checkpoint after signal15 cancellation; establish whether the suffix production arm ever began. No recovery or fit is authorized by this audit.',
'3. Patch focus: reconcile original GPU917001 and any exact pending-only relocation terminal. The current pin supplies CPU acceptance and relocation preparation, not a trained result.',
'4. Full image-teacher collection: verify one complete fitting-only collection terminal and aggregate count/binding if it exists; do not inspect individual teacher predictions or infer completion from1900/2247.',
'5. All-development native-public908454: final ten-epoch terminal and E9/E10 preservation fall outside the last observed state at this pin. Later evidence needs a new explicit snapshot.',
'6. Hard witness and slice relations: close the receipt-level engineering lineage if a fully enumerated runtime journal is needed. A completed gradient probe or small recovery is still not an accuracy trial.',
'7. Legacy unifiedtheory: obtain primary aggregate run receipts before admitting any README benchmark or claimed historical score. The archived provenance warning remains binding; raw clinical exports are not needed for this check.',
'8. Long-tail engineering: an exact all-jobs count would require an event-level manifest distinguishing preparation failures, allocated probes, deliberate resume stops, scheduler cancellations, private inference notebooks and scored submissions. The present family index does not claim that count.','',
'## Preservation rules','',
'Keep source/label identity corrections, conditional bootstrap support, weak-development versus public endpoints, original failed IDs, duplicate-control identities and interrupted states. A later success supplements the old failure rather than erasing it. The newer e4f288d6 focal-window module is outside this pin and has engineering tests only, not accuracy evidence.','']
(D/'experiment-family-coverage.md').write_text('\n'.join(lines))
with (D/'audit-entry-family-index.tsv').open('w',newline='') as f:
 w=csv.writer(f,delimiter='\t');w.writerow(['path','type','git_tree_or_blob_sha','name_based_category','family','content_review_claim','pinned_url'])
 for r in entries:w.writerow([r['path'],r['type'],r['sha'],r['category'],r['family'],r['content_review_claim'],'https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/'+('tree' if r['type']=='tree' else 'blob')+'/'+pin+'/initiative/audits/'+r['path']])
(D/'supplement-family-status.json').write_text(json.dumps({'snapshot':pin,'families':[dict(id=k,title=v[0],status=v[1],source=base+v[2],matching_entries=sum(r['family']==k for r in entries)) for k,v in F.items()]},indent=2)+'\n')
print('Family groups:',len(F),'protocol directories:',len(P),'metadata entries:',len(entries))
