# Current RSNA Knee research state

Updated October18 2026. The operational [agent guide](../AGENTS.md) is the current source of live owners, exact bindings and next actions. [Earlier snapshots](current-state.history-20261018-raw-renderer.md) retain historical outcomes and superseded pending states.

Best actual Kaggle: **0.958**, external R6 ensemble reproduction, submission57333150. Own influence full-CoAtNet result0.949; uniform0.947; parent0.949; independent ImageNet N3 0.894. Target0.97 has not been reached. Current model development uses the full73M-parameter CoAtNet, not a small DINOv3 head. No new paired/complementary-teacher accuracy gain is available.

The original paired experiment completed two epochs/972updates, with all model/optimizer/provenance objects privately verified. Its fixed-final third epoch139474998v1 is actually RUNNING; separate original scoring and R6-source controllers wait for actual final acceptance. Higher-LR E3 is its existing queued original, not a failed/restarted run. All five daily submissions were used at the latest authoritative allowance check.

The complementary-teacher input blocker is solved: missing source CSV interfaces were restored from original metadata, and the unchanged raw-DICOM/anatomy renderer reproduced **all248 MRI windows and metadata bytes exactly** without repeating the67 source downloads. Rendering took19.9647s for the smallest-source reference study; this is not a full-cohort runtime estimate. [Actual proof](../initiative/research/S6-HF-raw-input-bridge-recovery-20261018-v2/actual-render-result.json).

The complete fitting-source index is actually retained privately in Newton RAM:3887studies/720851DICOMs/21429series, compressed53.2MB, with all source identities/series metadata/individual1GiB source caps validated. No501GB permanent image copy was made. [Actual proof](../initiative/research/S6-fitting-source-index-20261018-v1/actual-source-index.json).

**Next:** execute a bounded source-render/FP32-H100 teacher pipeline using the existing index, original assets and reusable original predictions; then construct the complete equal-rank supervision overlay and run the matched full-encoder comparison. The source-buffer helper is prepared/tested, but full-cohort teacher generation and resulting training have not started. Protected labels/folds remain unchanged.

The earlier dense spatial reader did not show an established matched gain; repeating that unchanged expansion is not the priority. The paired gradient-noise reduction motivates the running experiment but is not an accuracy result. [Research mechanism](../initiative/experiments/competitive-v1/s6-rank-transport-v1/README.md).
