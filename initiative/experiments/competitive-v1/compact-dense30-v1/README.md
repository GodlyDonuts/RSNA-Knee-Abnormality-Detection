# Dense30 inference-only coverage contrast

Root implementation preserves every original matched-source-neighbor window and
its exact donor triple. Extra centers require both original source geometry edges;
cap30centers/series and six selected series,180windows/study. Farthest physical
depth fills coverage deterministically; equal distances choose lower source index.
No extra centers are invented without usable geometry. Legacy fallback triples
remain unchanged. Original-window indices permit direct bit-parity checking.

Images retain original source hashes/intensity/spatial mapping. Image-only dataset
never reads labels/masks. Evaluation refuses training-mode modules, unbounded
encoder chunks, incomplete rosters and hidden model failures. The policy is
separate from original inference; it is not a training sampler or a silently
changed deployment. Test expansion on the same checkpoint before any promotion.

Local synthetic checks cover source subset/parity/caps, short/missing/gapped/invalid
geometry, original bytes tampering, label non-access and explicit model failures.
Actual MRI admission and target-device throughput remain required. More windows
do not establish gain or nine-hour hidden completion. Do not combine with TTA.
