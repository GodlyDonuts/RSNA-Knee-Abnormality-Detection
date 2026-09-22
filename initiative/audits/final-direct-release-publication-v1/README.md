# Verified report-supervision data release

Training-ready data: **true**. Competitive performance established: **false**.

Git `f41879a33b38e47f0cbd96025edac7e2176a0a2f` (tree `aed8cda7cbb394bf665319a52073ab8926a16a10`, CI 35745491694) supplies all 17 unchanged payload files. The 11 release files are directly downloadable here; six supporting files are under `companions/`. `PUBLICATION.json` binds every payload hash, source identity, dependency seal, diagnostic acceptance and immutable complete checkpoint revision `c0b2c8b041120ae6d27891145f1a19e48a6d498f`.

Use `studies.parquet` and `study_labels.csv` with their released observation/training masks and exclusions. Missing or uncertain evidence must not become a negative. Preserve group boundaries and fixed splits; the 58 historically exposed official-gold studies remain evaluation-only. This is report-derived weak supervision, not perfect clinical labeling.

For runnable training and validation, restore the complete Git `f41879a33b38e47f0cbd96025edac7e2176a0a2f` tree from GitHub or the pinned base-plus-delta HF checkpoint listed in `PUBLICATION.json`, then place these files at the original `git_path` mappings in that manifest. The unchanged release receipts intentionally retain their original repository-relative references and historical publication status. Full original/reviewed annotations, policies, code, diagnostic output checkpoints, logs and provenance are preserved in that complete tree; this compact folder alone is not a standalone runtime installation. Follow the pinned `docs/data-readiness-report.md`, `docs/releases.md` and training strategy.

The root train/test image trees and official CSVs are referenced at source HF revision `0b5e410288232f570c69ba4f44723356e6ae6ab7` without modification. Cache `series-u16-v3` uses side 256; the accepted local arrays totaled 82,555,132,978 bytes (24,386 series, 819,635 slices). Those arrays are **not included** in this compact promotion: the index and effective inventory support rebuild/verification from pinned images. Rebuild commands and measured requirements are in the complete pinned documentation.

Actual short diagnostic acceptance covered optimizer updates, exact CPU resume, zero masked gradients, 58 gold outputs and 3 supplied test outputs. Competitive training, evaluation uncertainty, full hidden-test runtime and model ranking remain separate research tasks.

No root dataset card, source object or existing version is replaced. The final publication receipt records the immutable promoted revision and full streamed verification of all promoted bytes.
