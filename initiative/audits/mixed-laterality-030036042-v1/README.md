# Source and header assessment: 030:18, 036:17, 042:11

All three complete reports were personally read and frozen before case-specific
metadata was opened. The original blinded reports exactly match both the
fresh official and archived source CSV copies. Cases 030:18 and 042:11 are
brief unsided knee reports with complete technique/results/impression structure.
Case 036:17 explicitly describes the left knee in one continuous report and
summary. None contains a separate contralateral report block.

The assessment binds 19 series, 507 source slices and 38 existing lexical
endpoint headers. All endpoint positions, orientations, pixel spacings and
native dimensions match the corresponding full-cache metadata. Every source
path/SOP/hash membership and cache metadata hash matches the corrected inventory,
full audit and preserved recovery lineage. All 19 recorded image-field centers
were independently recomputed. No new network or pixel reads were performed.

Cases 030:18 and 042:11 have five-series protocols with R sagittal/axial tags
and an L/R coronal contrast pair. Paired coronal orientations and physical field
centers match despite different native matrix resolutions. Case 036:17 has an
eight-series L-tagged main protocol with LT-prefixed descriptions, followed by
one R-tagged supplementary coronal series. Its orientation and field center
match the main coronal pair despite different slice count and matrix size.

`source-header-assessment-and-review-plan.json` preserves exact series UIDs,
source/report hashes and pairwise geometry. Case 036:17 is the next priority
for bounded cached views because the report explicitly names left while one
supplementary series carries an R tag. No named-side correction or second-knee
claim follows from the metadata. Clinical annotations and gold were not opened.

The collector and geometry comparison require new outputs. Reproduction should
use a new sibling staging directory with the scripts and preserved source-only
preread, leaving this completed stage unchanged. Image-field centers account
for pixel spacing and matrix size; they are not anatomical landmarks, real-world
identity proof or a common registered frame across distinct anonymized frame
identifiers. Pixel review remains pending a separate bounded assignment.
