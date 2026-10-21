# Exact selected-frame MRI preparation

The actual descriptor-batching CPU comparison reduced warmed preparation11.46%,
but assembly still consumed about19.5seconds summed over three public studies.
A local profile attributes most remaining assembly work to full-volume finite
checks, masking/copying, and interpolation. This experiment keeps the original
selected24windows per mask while avoiding masked copies of unused source frames.

The complete source volume, aliases and descriptor cache are admitted once.
The identical canonical roster, centers, within-acquisition neighbors, original
metadata, every descriptor region and complete sparse supports are retained.
Only the intermediate image volume is lazy: the exact original checkerboard
and pixel halo are applied to each selected three-channel window before the
original bilinear interpolation. Duplicate frames use their first occurrence's
mask even when the duplicate's acquisition/ordinal differs. No source regions,
descriptors, support mass or chosen context windows are removed. All learned
models, original numerical modules and live training/serving remain unchanged.

Twenty-five scoped tests check complete byte equality, both original B/C
probabilities with nonzero evidence heads, odd matrices, aliases across
acquisitions, halo widths, context caps1/7/24/94, original RNG/source immutability,
and rejection of invalid *unused* frames and caches. Local synthetic assembly
on72canonical frames fell0.815/0.819→0.402/0.326seconds;192frames fell
2.073/1.991→0.541/0.570seconds. This is CPU/PyTorch2.14 synthetic evidence,
not real-MRI throughput, Kaggle/T4 acceptance or accuracy.

The distinct private CPU-only Kaggle notebook alternates original/selected
assembly/original/selected/original. Both arms use the same previously verified
batched descriptor helper. Every complete context byte and decode statistic
must match. It measures descriptors, assembly, decode, total wall time and
process-tree memory. Three public studies and sequential cache/host effects
remain limitations. No model, labels, Dataset, GPU quota or entry are involved.

Source140038565v1/d55da863 anchors every unchanged numerical module and helper;
only the benchmark and new selected-frame assembler are added. The coordinator
uses one exclusive canonical account claim, one private notebook version and
no uncertain write retry. New8MiB node controls extend the two prior CPU
experiments'24MiB to32MiB, alongside516GiB other-node liabilities and20GiB floor.
Shared123MiB/370files plus20GiB/5000reserve and the serialized existing claim
are retained. No deletion credit. Actual push/terminal/timing must be recorded
separately; implementation alone does not authorize a production change.


Actual private CPU-only Kaggle140041310v1 completed, sourcee5385a8dc72e,
CPU998715 terminal collection. Every complete original context byte matched.
Warm two-thread wall means31.842→22.944seconds (27.94% less); summed assembly
17.362→4.707seconds (72.89% less). The first cold reference is excluded. Both
arms already used the same exact descriptor batching; gains from different
CPU hosts must not be multiplied into a claimed measured combined speedup.
See [actual aggregate receipt](actual-Kaggle-CPU-result.json).

A preparation-only serving successor is implemented and34 scoped tests passed.
It retains every original model/decoder/precision/CSV path and requires full
original-versus-optimized context byte equality on all three studies in its own
trained two-T4 qualification. Production qualification is not yet executed.
The additional CPU comparison is inside the already-funded qualification,
not a new GPU notebook or submission allocation.
