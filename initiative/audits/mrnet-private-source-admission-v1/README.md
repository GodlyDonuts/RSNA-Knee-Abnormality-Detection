# Stanford MRNet original-source admission

The owner explicitly directed Stanford use on October 1. The approved Redivis
account now has legitimate read-only `data.data` OAuth access; the token is only
in the ignored private local `.env` and streamed through SSH stdin. No refresh
token or personal application answers were archived.

All eight original v1.0 API CSVs were retrieved once to ignored private local
storage and their exact hashes checked. `schema-admission.json` records the
source version, observed memberships, counts, and original byte proofs without
republishing the source label values or images. The 3,390 training volumes cover
1,130 exams; only 1,129 have all three observed labels. Validation has 360 volumes
and 119 observed labeled exams out of 120. The imports omit training case 0000
and validation case 1130. Their labels are not reconstructed from column names.

`scripts/newton/ingest_mrnet.py` accepts the exact original memberships and source
checksums. Its private CPU acquisition verifies every file's original MD5, byte
count and SHA256, then validates a primitive uint8 slice-stack NPY envelope
without loading a pickle. Original volumes remain verbatim on private node-local
storage. It reserves no GPU, preserves a 20-GiB node-local reserve, and has no
automatic mutation retry. The single coordinator refuses an existing intent or
control until reconciled read-only. Shared metadata is bounded to 64 MiB/160
files with native and sampling liabilities also deducted.

The source tasks are **ACL tear, generic meniscal tear, and abnormality**. These
are source-specific auxiliary tasks, not a substitute for the twelve RSNA
targets. Generic meniscus is never mapped into both medial and lateral targets;
abnormality never creates twelve positive labels. Source validation remains out
of training. Arrays are already processed 256-pixel volumes, not native DICOM;
physical positions and unobserved protocol flags remain unknown.

The intended experiment is private MRI-domain adapter warm-up with three separate
heads, transfer of only the compatible encoder adapter into the RSNA model, and
the same RSNA final-fitting recipe as a matched compute-budget control. Evaluate
all twelve RSNA targets on the unchanged development pool. Reject the transfer
if the paired held-out result does not improve or harms sparse targets; additional
data and Stanford-task accuracy alone are insufficient evidence. Patient counts
are not inferred from exam counts, and cross-source duplicate screening is needed
before attributing a clean held-out gain to external pretraining.

The accepted access agreement permits personal noncommercial research and
restricts redistribution and derivatives. This lane currently performs verbatim
private research acquisition. Competition and model-weight redistribution
compatibility remain unresolved; raw Stanford files are not uploaded to GitHub
or Hugging Face. Do not claim that attribution alone resolves those terms.

At this source milestone, acquisition code and schema acceptance are tested;
images have not yet been acquired and no Stanford model fitting has run. Exact
dispatch and completion receipts will supersede this statement as they arrive.

Sources: [official MRNet description](https://stanfordmlgroup.github.io/competitions/mrnet/),
[Stanford Redivis v1.0](https://stanford.redivis.com/datasets/4a2c-4cpkzrn2c),
[Redivis file API](https://docs.redivis.com/api/rest-api/files/get).

## Actual acquisition completion

Original CPU-only job **867378 COMPLETED 0:0**, elapsed 10:18 on evc32.
All **3,750 volumes / 7,740,871,424 bytes** passed their original MD5, byte count,
SHA256 and primitive uint8 NPY envelope checks. The original acquisition/control/
source/scratch/terminal receipts are in `acquisition/`. The full private volume
journal SHA is `4cd2522fcd6b99bac778edf2f72be24fb7d0aca497a1f0f0e882412646515dbc`;
its raw images and original label tables remain on the owning private node.
Do not repeat this acquisition or move the raw volumes into shared quota.

`rsna/competitive/mrnet.py` now supplies exam-level three-plane source tensors,
observed ordinal array indices (not invented physical spacing), separate three-
task attention and adapter-only transfer. Original label-table bytes are checked
before admitting derived labels. Padding cannot reach the encoder or introduce
NaN gradients. Tests verify validation separation, label tamper rejection,
exam-loss gradients into adapters and atomic transfer rejection for a mismatched
base. These are engineering checks, not accuracy evidence.

The real-source CPU tensor probe is prepared in `probe_mrnet_inputs.py` and its
one-CPU pinned-runtime wrapper `launch_mrnet_inputs.py`. Its purpose is to decode
eight actual source exams, bind source labels and tensor hashes, and confirm the
full 1129/119 split membership. No encoder, optimizer, clinical inference or
Stanford warm-up has yet run. Require actual input receipt before fitting.

## Actual tensor admission and bounded research pilot

CPU job **867399 COMPLETED 0:0**, 30 seconds on evc32. It verified all 1129/119
supervised memberships and eight actual three-plane exam tensors at 256 pixels,
with the full 16-slice-per-plane budget. See `input-admission/`. The full original
per-case receipt stays on private cluster storage; its SHA is
`146197f268471e167b2e16fb5e0724b24a58b714d3882567ec0bc09d9c95caf1`.

The bounded source-specific adapter recovery driver now uses the pinned official
DINOv3-L, rank32 and a separate three-task exam head. It runs sixteen training
exams for two epochs/four effective-eight updates, comparing uninterrupted and
fresh-process interrupted/resumed trajectories. Four source-validation outputs
are engineering predictions only; no diagnostic AUC selects the recipe. Require
exact learned/base/AdamW/LR/counter/RNG equality and identical prediction bytes.
All learned states remain on private node-local storage under a separately funded
2-GiB checkpoint bound, with the 20-GiB disk reserve and maximum two GPU nodes.

This is bounded private noncommercial research under the source's stated research
purpose and the owner's explicit direction. Original dataset bytes are retained
unchanged. It does not establish permission to redistribute Stanford-derived
weights, nor competition compatibility. No such weights may be mirrored or
submitted without resolving that downstream boundary. The source-specific head
is never transferred into RSNA targets. No GPU pilot has been dispatched at this
preparation milestone; later exact dispatch receipts supersede this statement.
