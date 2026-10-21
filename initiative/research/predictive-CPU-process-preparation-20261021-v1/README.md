# Original MRI preparation: thread/process comparison

The exact C public receipt140024153v1 spends111.8 summed study-seconds in
DICOM decoding/rendering versus6.57 in GPU inference, approximately94% in
preparation. This motivates separating CPU preparation from GPU inference.
It does not prove a process speedup or hidden nine-hour success.

This bounded experiment uses only exact accepted C input modules from source
`c414ecfe6d7fff62c1f3b68a19f4874a79a6bb28e00a32df1917c3da592d6509` and its
pinned Kaggle Docker image. The three public MRI studies run thread2,
process2,process3,thread2,process2. Every image/metadata/target/mask/support
tensor and structured value must remain byte identical, with original decode
statistics unchanged. Timings include spawn, imports, packing, IPC, hashing
and shutdown. Process-tree RSS is capped24GiB and total work900seconds.
No model, clinical label, prediction or submission CSV is read/produced.

One private offline **CPU-only** notebook is authorized through the canonical
Newton ledger/claim, no new Dataset, GPU allocation or competition entry.
Control CPU uses one core/zero GPUs on ordinary evc27. New node allowance8MiB,
existing other node liabilities516GiB,20GiB floor; shared123MiB/370file
existing liabilities plus20GiB/5000file reserve, no new shared scope.
Existing intent or uncertain response forbids another push. Fresh quota is
checked before account work. A local synthetic DICOM test actually exercises
both threads and spawn processes with complete original contexts; API tests
reject altered source, GPU allocation, wrong tensor hashes and runtime/memory
overages. The initial64-pixel synthetic fixture was invalid at halo8; it was
corrected to96x112, without changing numerical code.

Sequential trial order/page-cache effects and the tiny public roster limit
interpretation. A useful CPU result still requires a separate exact two-T4
pipeline acceptance before production use. Healthy B/C/C0/C100/Colab science
and existing submissions remain unchanged. The conditional C+R6 blend remains
unadmitted on runtime; this experiment does not establish accuracy.
