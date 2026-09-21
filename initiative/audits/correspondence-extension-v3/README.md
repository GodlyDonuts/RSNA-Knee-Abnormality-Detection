# Nine-case correspondence exclusion proposal

Prepared by `/root/competition_contract`; not published or committed. The production initiative and required registry remain unchanged.

The staged overlay adds nine source-bound exclusions to the four existing exclusions (13 total). The runtime change is limited to `rsna/correspondence.py`; the new tests are `tests/test_correspondence_extension.py`. No `labels.py` or other shared production code was edited.

## Schema

The policy retains schema version 1 and adds `proof_binding_version: 2`. This requires every record to use `verified_multi_assessor_correspondence_source_binding`; v1/v2 policies keep their current validation path. Each record retains twelve false report masks and binds its original blinded batch and UTF-8 report hash.

Each new proof contains: schema version; actual normalizing agent/session; case and study UID; exact source batch, zero-based source index and report hash; literal source quotations; `root_decision` reference and `root_selector`; an exact `{pointer, value}` root disposition; independent assessment reference and exact supported assessment claim; and either `reviewed_native_aggregate` or `reviewed_cache_aggregate`. Cache proofs additionally bind the original source handoff.

Runtime validation checks exact source identity and quotes, root authorship, a distinct independent assessor, case/study identity, all referenced file hashes, and original explicit exclusion content. Cache roots must select a case with `required_report_supervision_mask_all` exactly true, bind the same independent assessment and handoff hashes, and preserve its source report hash. Older native decisions use an exact allowlist of the three immutable historical mask decisions or the explicit preserved 067 quarantine/action pair, plus original source and independent-evidence bindings. A generic verified status, arbitrary nonempty identity claim, literal tag or clinical flag cannot substitute.

Original annotations, evidence, target values, splits, leakage groups and official evaluation masks remain unchanged. This is report-supervision exclusion, not a claim of image diagnosis, named-side correctness, authenticity, or completed release readiness.

## Evidence anchors

| Case | Source report SHA-256 | Preserved authoritative root proof | Independent proof |
| --- | --- | --- | --- |
| 017:07 | `b31b75ef8fa55cecdea6bfe39a27b6ac7b7af4ec0ca290e93428f156293b2b46` | `initiative/audits/bilateral-017-007/decision.json` selector `(root)` | `initiative/audits/bilateral-017-007/independent-image-review.json` |
| 018:19 | `f93db6aa92712f84b6362cae5757923d4b74e1e77cd08bd0e705aed9250ce139` | `initiative/audits/bilateral-018-019/decision.json` selector `(root)` | `initiative/audits/bilateral-018-019/independent-image-review.json` |
| 038:37 | `573db85d7683a63e19b6864e96d3dce06616b4909decd7018e817874eeae2e82` | `initiative/audits/bilateral-038-037/decision.json` selector `(root)` | `initiative/audits/bilateral-038-037/image-review.json` |
| 067:37 | `11f7d4cc72aadec83fb4ecd6eba8047f81d06e1efa3723635543c3191cd411d0` | `initiative/audits/bilateral-067-037/decision.json` selector `(root)` | `initiative/audits/bilateral-067-037/independent-image-review.json` |
| 052:01 | `db1dfc9384c5f16dc4654549fb40f0ed2fe998e43482d0f118666da579f89dec` | `initiative/audits/correspondence-052001-017-cache-v1/root-source-geometry-display-review.json` selector `/cases/0` | `initiative/audits/correspondence-052001-017-cache-v1/cache-panels/case-052-01/independent-cache-visual-assessment.json` |
| 052:17 | `6dedb30fb3cf6bf742ebc83d14ebf6c681bbb64cfd27115bab6631e1330583d3` | `initiative/audits/correspondence-052001-017-cache-v1/root-source-geometry-display-review.json` selector `/cases/1` | `initiative/audits/correspondence-052001-017-cache-v1/cache-panels/case-052-17/independent-cache-visual-assessment.json` |
| 001:39 | `83936e2ff5f617ac5a90153b5a1bd183955952972e86c5f3f74c6896e6a3d4a9` | `initiative/audits/technical-tier4-bilateral-001007-cache-v1/root-source-geometry-display-review.json` selector `/cases/0` | `initiative/audits/technical-tier4-bilateral-001007-cache-v1/cache-panels/case-001-39/independent-cache-visual-assessment.json` |
| 007:16 | `37899fa13481f88c4794e5266aec9f1a965c1d892dadb1a647f2437c527f7975` | `initiative/audits/technical-tier4-bilateral-001007-cache-v1/root-source-geometry-display-review.json` selector `/cases/1` | `initiative/audits/technical-tier4-bilateral-001007-cache-v1/cache-panels/case-007-16/independent-cache-visual-assessment.json` |
| 062:21 | `9aeb3bd3503ae2a7796f8ccf7af877f81d65e76ceed735a9b47d0ffe3d0d8b59` | `initiative/audits/correspondence-055058060062063-v1/root-source-geometry-display-review.json` selector `/cases/4` | `initiative/audits/correspondence-055058060062063-v1/cache-panels/case-062-21/independent-cache-visual-assessment.json` |

All full SHA-256 references and original source paths are recorded in `preparation.json` and the nine immutable staged wrappers. Full source text was personally read before preparing these structural bindings. The first four cases rely on preserved native anatomy/orientation assessments of two acquisition sets; the other five preserve unresolved correspondence between bilateral reports and bounded sampled protocols. Sparse cache evidence does not prove whole-study laterality or authorize selecting a source block.

## Review and promotion

Review `production.patch`, `dedicated-tests.patch`, `preparation.json`, and `overlay/initiative/audits/`. The overlay contains only the candidate registry, new v3 policy, and nine new proof wrappers. Preserve the old v1/v2 policy files unchanged. Check that the production registry still matches `base_registry.sha256` before any later root-authorized promotion; merge instead of overwriting if it changed. Do not run `prepare.py` a second time over frozen outputs.

Focused command (from `work/recovered`):

```sh
RSNA_CORRESPONDENCE_OVERLAY="$PWD/../annotation-staging/correspondence-extension-v3/overlay" python3 -m pytest -q tests/test_correspondence.py tests/test_correspondence_extension.py
python3 -m ruff check rsna/correspondence.py tests/test_correspondence_extension.py
python3 -m ruff format --check rsna/correspondence.py tests/test_correspondence_extension.py
../../.venv/bin/python -m mypy rsna/correspondence.py
```

Tests use fabricated target values and a fabricated gold assignment only; they never read official target columns. They cover exact real policy/source membership, all-target report masking and unchanged labels/groups/gold behavior, wrong source/index/quote/case/selector, downgrade to a generic proof, identity masquerading as assessment content, missing references, wrong proof kind, non-exclusion root decisions, same/wrong assessor, wrong study, altered claims/unbound evidence, and inner source mismatches after outer hashes are recomputed. The dedicated suite uses the staged overlay until promotion, then discovers the actual v3 policy at the repository path.


## Identity hardening after root review

Assessor identities now must be canonical task IDs, with no leading/trailing whitespace, empty path segments, whitespace within a segment or other path aliases. The canonical root, independent and normalizing identities are compared before accepting distinct roles. Regression cases rebind all outer digests and the root independent-artifact digest, so an alias cannot fail merely because of an unrelated checksum mismatch. Additional tests replace wrappers with only stage/status/case/UID/author metadata, trying both the new stage and legacy/generic stages; binding-version 2 rejects every downgrade.

## Exact validation boundary

For the five cache cases the hook recursively opens and verifies the root decision, selected case, independent assessment and source handoff, including the root decision's matching independent/handoff hashes and the handoff's exact report hash and assessor role. It **does not recursively open the earlier root frozen visual assessment, source-only prereads, panel images or the handoff's nested role/order bindings**. Nor does it rerun native/cache/display verification. Source-first ordering and the underlying visual observations remain assertions of the preserved hash-bound root/independent proof records. This scope is intentional and must not be described as a new independent replay of those earlier checks.

The nine staged wrappers/policy/registry remain unchanged. Latest code and test diffs after this hardening are `production-identity-hardening.patch` and `dedicated-tests-identity-hardening.patch`; `validation-v2.json` records the final checks. The first diff/check records remain preserved for review history.
