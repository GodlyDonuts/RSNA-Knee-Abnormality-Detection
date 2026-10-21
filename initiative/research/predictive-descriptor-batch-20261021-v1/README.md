# Exact batched MRI descriptors

October21: the first thread-versus-process experiment only improved warmed
three-study input preparation6.17%, with order/cache confounding. It did not
justify changing healthy serving. This distinct experiment targets the
thousands of repeated small descriptor operations within that preparation.

The helper batches CPU FP32 pooling, gradients and texture operations for up
to16 equal-size tiles. It retains the original per-patch8x8 matrix products,
DCT order, pixel scaling, alias membership, masks, output ordering, descriptors,
source priors and complete contexts. No learned graph, weight, label or fitting
recipe changes. Original inputs.py/cases.py/supports.py are embedded unchanged.
Stacked tile pixels are capped4MiB; unusually large tiles use the original
scalar implementation, with no discarded input or fallback prediction.

The actual local72-frame synthetic comparison includes original frame
validation and output packing. Reference0.755/0.714s versus batched0.593/0.601s,
all descriptor bytes identical. Local PyTorch2.14.0/one CPU differs from Kaggle;
this establishes neither real-MRI speed nor hidden-runtime success. Eighteen
scoped tests cover odd/noncontiguous matrices, boundaries, alias conflicts,
complete two-mask inputs/supports, unchanged sources/RNG and invalid data,
plus private CPU-only request and receipt acceptance.

The new Kaggle experiment alternates reference/batched/reference/batched/reference,
with two preparation threads and identical complete original contexts required.
It records decoding, descriptors, assembly, total wall time and process-tree
RAM separately. Three public studies, sequential cache effects and CPU/GPU-host
differences limit any performance claim. A full production inference change
would still need actual two-T4 model-output/runtime evidence; no deployment
change follows from implementation or this CPU experiment alone.

The original CPU notebook140034892v1/sourcea61ab785 and accepted C
140024153v1/sourcec414ecfe anchor the unchanged modules/runtime. This new
private offline notebook has a distinct slug/root/ledger key; no Dataset,
model download, GPU allocation or competition submission. Exactly one push
with preserved intent, canonical exclusive claim and no uncertain retry.
The one-core/zero-GPU account coordinator runs on ordinary evc27. Fresh
storage must retain516GiB other node liabilities plus24MiB for the original
16MiB CPU controls and this8MiB control, then20GiB reserve. Shared123MiB/370
file obligations and20GiB/5000reserve remain; reuse the serialized claim.
No existing original is deleted or its charged bytes credited twice.

Actual private CPU notebook **140038565 v1**, source `d55da8636a5b`, is
**COMPLETE**. Original CPU998426 pushed once; read-only CPU998521 collected
the exact acceptance. All five trials preserved complete input/context bytes.
The first cold reference took46.016s. The remaining reference runs took41.207
and43.415s; batched runs took37.164 and37.761s. Warm mean wall time therefore
fell42.311→37.462s (**11.46%**). Summed descriptor time fell20.980→12.761s
(**39.17%**); decoding and assembly remain material costs. Peak process-tree
RSS was4,302,151,680bytes, and total experiment time205.952s.

This accepts the CPU preparation helper on these inputs, not a production
GPU pipeline, hidden-runtime guarantee or accuracy improvement. Sequential
cache/host effects remain. Existing C100/Colab consumers and submitted
B/C/C0 notebooks are unchanged. Protect their allocated GPU quota; require
a separate exact two-T4 output/runtime decision before using this helper.
No GPU quota, Dataset or competition entry was consumed. Original private
receipt identity, module hashes and sanitized aggregates are retained in
[actual-result.json](actual-result.json); no MRI pixels or study IDs are published.
