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

## Actual result: modest warmed improvement, no production change

Private offline CPU notebook140034892v1 completed in132.0576seconds with
4.79GiB peak process-tree RSS. Complete input bytes matched in all five trials.
Thread2cold35.906s, process2cold26.025s, process3 24.352s, thread2warm23.397s,
process2warm21.952s. The warmed process pair was6.17% faster; trial order and
page-cache effects are confounded. No GPU inference overlap was measured and
CPU-only hardware is not the GPU host. This is insufficient to justify spending
the protected final-model GPU budget or changing healthy production inference.
The nine-hour hidden-runtime problem is not proved solved.

The first push returnedHTTP400 and was reconciled absent by authenticated
owned-notebook listing and exact search, with a positive control. Original
CPU998076 and intent are retained; the exact400cause is unproven. The separate
request-v2 used source5ffe6a9a and distinct `rsna-predictive-cpu2-a61ab7856149`,
matching title and pinned Docker request fields. CPU998135 returned the actual
version once; CPU998187 observed its completion. Fresh storage funded16MiB
for both controls, preserving other liabilities/reserves. No competition entry,
GPU-VM time or model Dataset was consumed. [Sanitized actual result](actual-CPU-comparison.json).
