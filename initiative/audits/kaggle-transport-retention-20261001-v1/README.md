# Fund further solo experiments without discarding model payloads

CPU-only job **866637** completed **0:0** on evc21 in 25 seconds. It removed exactly
two redundant Kaggle transport ZIPs totaling **2,800,070,372 bytes** (2.61 GiB).
All **91 unpacked payload files**, including both checkpoints, the official
base weights, code, licenses and manifests, remain on private cluster storage.
All training finals, epoch sidecars, publication journals, accepted data and
private cloud objects remain intact. No Dataset create, notebook push or
competition submission was made.

Before deletion, the one-CPU/zero-GPU process verified each whole ZIP hash,
every ZIP member's size/hash, and every retained unpacked member's size/hash.
Both original private full-download receipts matched the exact payload lists.
Fresh authenticated GetDataset checks confirmed dedicated private version 1
Datasets **12299834** and **12293124**. The canonical Kaggle coordinator claim
excluded concurrent publication during validation and eviction. A durable
finite-list eviction intent was recorded only after every copy passed; the
terminal read confirmed both ZIPs absent, all 91 members present and lock released.

The first CPU check **866636** failed before hashing or any eviction intent:
the authorized `/home/sa305415` root resolves to `/lustre/fs1/home/sa305415`.
Both ZIPs remained present. The distinct v2 invocation compares resolved paths
against the resolved authorized root, rejects internal symlinks, and uses a
new control. Original source, failed dispatch and read-only failure proof are
preserved beside the corrected source and actual completed receipt. Do not
repeat either invocation.

`plan-v2.json`, `verify_and_retain_v2.py.txt`, `verified-v2.json` and
`terminal-v2.json` record the exact operation. The existing source/cache jobs
and queued sampling pilot were unchanged. Fresh native admission projects
22.846 GiB and 5,162 files after its existing obligations; the additional sampling
pilot's 145,584,750 bytes and 50 files must still be subtracted. This frees capacity;
it does not by itself admit another production job or establish model accuracy.
