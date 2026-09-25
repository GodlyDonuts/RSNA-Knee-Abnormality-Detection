# Fold-2 DINOv3 pixel-payload deployment attempt

The owner authorized up to the live Kaggle allowance before the 2026-09-25 UTC
boundary. Authenticated `GetSubmissionLimits` at 2026-09-24 17:53 UTC reported
two submissions today and three available slots. Neither resource preparation
nor a public notebook run is a competition submission.

The candidate projects unchanged baseline training code from
`b74060b98f01bff1b924070ccba5ce44dd002f40` and the packaging workflow
from `df0e283aaaea4a10d77edf6633d598ac4fcb88db`. The only deployment
overlay is `scripts/kaggle/chunked_infer.py` from
`e89f604419f2159bbc20105761c0acd0efd16dbb`, SHA-256
`8d20d1125c8cc886b7090aee35d210fb94b1d5f2835378e449265ae7fdb91b28`.
It catches pydicom's truncated pixel-payload error and allows another valid
same-plane MRI series. Projection manifest SHA-256 is
`2f1de61a12eb3fe5f53f56ce6cc3098c23abd9c965669e87798ece2fb3abae75`.
The narrow regression passed against a synthetic truncated DICOM, but a hidden
test decoder failure remains a hypothesis.

The first prepared candidate used fold 2 epoch 10, checkpoint SHA-256
`5e400e82c4b50e94a27efcb250fc86d911b2191b6ed04792276fe51d9e093734`.
Its private Dataset `godlydonuts/rsna-knee-f2e010-pixelfix-5e400e82c4b5`
version 1 was created once and fully downloaded/hash verified. Its notebook
push returned an ambiguous `RemoteError`. The canonical Newton ledger records
`push_uncertain`. An authenticated `GetKernel` returned HTTP 403 and exhaustive
owned-profile listing showed no matching slug. This is still an uncertain
mutation: never retry that notebook slug or submit this candidate without a
separately reviewed recovery.

A distinct eligible checkpoint, fold 2 epoch 8 SHA-256
`eb280864bd1c2f896210ce360e1ae36d59649c854a9051523c2454a34d08c7b1`,
was packaged without changing training code or inference recipe. Its private
Dataset `godlydonuts/rsna-knee-f2e008-pixelfix-eb280864bd1c` version 1,
ID `12173582`, was created once and every file downloaded/hash verified. The
distinct private offline T4 notebook
`godlydonuts/rsna-knee-epoch8-f2-pixelfix-eb280864bd1c` version 1 was pushed
once; the returned exact source SHA-256 is
`a791eeaa2cad9834b7abb5cce9c86b41496f688c8b382add23c1bc8c03b2244a`.
It then `COMPLETE`d its MRI-only public execution. The exact
[public-example receipt](public-example-receipt.json) bound the checkpoint and
bundle manifest to three studies. The generated
[public-example-only CSV](public-example-only.csv) had exactly the three
official study IDs in order, 13 expected columns, finite probabilities in
`[0,1]`, and SHA-256
`a3d29bc7f8c6908b6db13162bb495e76623f837c3caa4d3bb40067250acf0350`,
matching the receipt. This CSV was inspected, **not uploaded**.

After a fresh authenticated limit check still showed three slots, the single
Newton coordinator submitted that exact notebook version once with Kaggle's
`CreateCodeSubmission` API, file name `submission.csv`, and received
submission **56528793** at 2026-09-24 18:19 UTC. The first
[live observation](submission-first-observation.json) recorded no public
score or terminal hidden-run result and two remaining live slots. A successful
public three-study notebook is not evidence that the hidden rerun will score;
hold further competition submissions until this outcome is known or a new
decision is justified by the live account window.

The one authoritative account ledger remains
`/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3`. Large checkpoint,
runtime and Dataset artifacts remain on private Newton storage. Credentials
remain in the ignored local `.env` and are streamed to bounded CPU-only Slurm
coordinator phases over SSH stdin. Each mutation has a durable intent; an
uncertain outcome must be reconciled read-only, never retried blindly.

**Hidden outcome:** Authenticated Kaggle `ListSubmissions` subsequently marked
submission 56528793 `COMPLETE` with zero output bytes, an unhandled-notebook
error and **no public score**; see the [exact read-only receipt](hidden-outcome.json).
The underlying hidden traceback is not available, so a corrupt test series or
larger runtime is a hypothesis, not a proven cause. Do not resubmit this
version or send another checkpoint through the same inference path. Two live
slots remained at observation. The fold-2 epoch-6 backup was packaged and its
private Dataset was created once, but full remote checkpoint-download
verification encountered intermittent Kaggle API errors and a broken Mac-side
transfer; no epoch-6 notebook or competition entry has been created. Its
Dataset create must never be repeated. The distinct fold-2 epoch-10 notebook
push remains uncertain and its old slug must never be retried.
