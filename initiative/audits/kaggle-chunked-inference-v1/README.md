# Bounded Kaggle inference preparation

The offline code notebook previously decoded every hidden test study into one
temporary cache before model inference. The accepted training cache contains
82,794,366,615 unpacked bytes for 4,407 studies, or about 18.8 MB per study.
At this average, the organizer's approximate 1,300-study hidden set could
occupy about 24.4 GB in a monolithic temporary cache. Hidden DICOM sizes and
the actual Kaggle filesystem budget remain unmeasured.

`scripts/kaggle/chunked_infer.py` runs the unchanged trained inference code on
successive 32-study subsets. Each chunk's checkpoint SHA, diagnostic flag,
submission SHA, study count, output identities/order and model provenance are
checked before its decoded cache is deleted. The final submission is validated
against the full official template. The private Dataset bundle includes this
wrapper and verifies its bytes. `rsna/**/*.py`, accepted training masks/folds
and checkpoint bytes are unchanged. This packaging source must be projected
from its committed Git revision, with the exact training source commit bound
separately.

Local verification uses a diagnostic three-study synthetic checkpoint and its
prebuilt series cache: chunked output equals monolithic output on all 12 target
probabilities and preserves template order. Fake-inference tests cover a
cross-chunk duplicate SeriesInstanceUID and a misordered output failure. These
checks do not prove hidden DICOM decoding, T4 throughput, or a Kaggle score.
An eligible production checkpoint and exact private offline notebook version
are still required before any competition submission.
