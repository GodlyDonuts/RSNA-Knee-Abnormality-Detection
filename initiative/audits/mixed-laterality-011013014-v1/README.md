# Mixed laterality triage: 011:03, 013:11, 014:19

All three complete original reports were personally read before their candidate-specific headers. Each is one unsided narrative; none explicitly describes separate right and left examinations. The source strings exactly match both the archived CSV and the preserved current authenticated official CSV. No clinical review annotations or gold values were opened.

Complete selected scope is 17 series, 497 inventory slices and 34 lexical first/last headers. Every selected header source identity, SHA and size matches the corrected inventory. Cache metadata matches its frozen recovery lineage and full source membership. All 34 endpoint positions, orientations, spacing and dimensions exactly match corresponding cache SOP metadata.

The observed conflicts occur inside a single apparent protocol: 011:03 has five L-tagged standard series and an R-tagged 15-slice oblique series; 013:11 and 014:19 have R-tagged sagittal/axial series with L-tagged coronal series. The additional 014:19 oblique series is R-tagged. Reuse of a generic sequence description across planes is not a repeated full protocol.

Recorded image-field centers were calculated from each complete series' mean image origin, native dimensions, spacing and IOP. They cluster within each study, supporting a one-protocol tag inconsistency candidate rather than separated bilateral acquisitions. These centers are not anatomical landmarks, and distinct anonymized frame IDs prevent asserting global registration or anatomical identity from numerical proximity alone.

The next useful step is a small source/cache-bound representative visual review across the conflicting plane families, including the oblique sequences. No full source fetch or pixel inspection was performed in this task. Do not automatically change tags, infer a named side, assign labels or claim a source institution.

`collect_technical.py` reproduces source, inventory, header and cache metadata evidence in a new output directory. `compare_geometry.py` independently verifies exact endpoint/cache geometry. `verify_image_centers.py` reproduces all 17 recorded image-field centers. `source-header-assessment-and-review-plan.json` gives exact series memberships and bounded follow-up recommendations.
