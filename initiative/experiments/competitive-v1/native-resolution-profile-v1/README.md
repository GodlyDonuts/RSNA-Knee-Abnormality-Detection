# Original matrix profile and separate native-detail experiment

September 30, 2026. `verified.json` measures the original archived first/last
DICOM headers for the accepted model-series roster. This is a metadata audit,
not a diagnostic reread, a new cache or a trained experiment.

The profile covers **3,403 usable development studies, 18,906 model series and
37,812 endpoints**. First/last dimensions agree for every included series.
**95.0439%** of these series have matrices larger than 256 in both axes. Common
matrices are 512×512 (4,613 series), 640×640 (2,523), 384×384 (2,183), 320×320
(1,465), 256×256 (881) and 1024×1024 (832). The full histogram is preserved.

The compressed and uncompressed header archives were independently rehashed
against their existing archive receipts. Study/series membership is bound to
the accepted parquet and series CSV hashes. The script reads only development
UIDs, split and observed-mask columns; target values, reports, gold labels and
reserved confirmation labels are not read. It excludes fully masked studies.

Replay without modifying the archive:

```bash
python3 initiative/experiments/competitive-v1/native-resolution-profile-v1/profile.py.txt
```

The current adapter inputs are 224 pixels from a 256-pixel cache. A larger
reconstructed matrix supports investigating native detail, but does not prove
acquisition resolution, visible lesions or the cause of the 0.769 Kaggle score.
These endpoint headers predate the two official encoding recoveries; a rebuild
must verify the currently pinned original DICOM bytes and every used slice's
geometry. Upsampling the existing cache cannot recover discarded information.

## Controlled next experiment

Keep rank, study membership, labels, masks, fold, optimizer, sampling and schedule
fixed. First prepare a bounded CPU decode pilot from pinned original DICOM
objects, covering differing matrix sizes and accepted sequence families. Require
finite correctly oriented pixels, exact slice membership/order, aspect-preserving
geometry, measured native-to-input transforms and cache hashes. Keep the current
256 cache intact. No labels need to change for this experiment.

Use actual 384- or 448-pixel inputs from the new decoded source as a distinct
comparison after the CPU and allocated throughput checks. Choose the final side
from measured memory, throughput and image evidence, not from matrix size alone.
Compare the same fixed-final outer fold and all twelve target supports; a lower
training loss is insufficient for promotion. Rank 64 and whole-series coverage
remain independent experiments.

An authenticated Stokes storage probe confirms that its home directory is the
same `/lustre/fs1/home/sa305415` quota used by Newton. Shared-filesystem free
space and login-node `/tmp` capacity are not an accepted private storage budget.
The current runs and deployments already consume their reserved capacity. A
complete higher-resolution cache has **not** been built or dispatched. Require
fresh byte/file quota and authorized allocated scratch, then measure a bounded
sharded or streamed design before committing to a full cache. See the separate
rank64/staging audit for the actual storage receipt.
