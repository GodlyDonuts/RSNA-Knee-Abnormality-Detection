# Original-source MRI input bridge recovery

Original CPU961339 stopped before native rendering because the directory adapter also requires `train_series.csv` and `train.csv`. All67 original DICOMs (25131360bytes) had passed size/SHA and identity checks; they remain in the original owner-only RAM directory. No acquisition is repeated.

This distinct recovery supplies only the three matching original series-metadata rows from `initiative/source/train_series.csv` (whole-file SHA573c1d80772bf41211c91b149c95677385a1c22d63f485c347f1b46c0177aef3) and an ID-only one-study roster. No diagnosis/report values are read. The frozen native/anatomy/window code, size320/viewsv4/stride1/physical3.5 geometry and original reference tensors stay unchanged. Success still requires exact image and metadata tensor bytes.

One zero-GPU2CPU/8GiB/12minute CPU job, same original1GiB RAM cap. The final2MiB small-control reservation uses the remainder of the original16MiB S6 envelope; no disk/data/model reservation is enlarged. Fresh original resource guards remain. A new failed result is preserved, never automatically retried. No full-cohort teacher or accuracy claim is authorized by this single-study bridge.
