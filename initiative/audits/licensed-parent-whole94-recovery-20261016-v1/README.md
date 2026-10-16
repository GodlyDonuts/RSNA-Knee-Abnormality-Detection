# Whole-study94-window MIL with bounded activation memory

Prospective extension, October16. The original12-window numerical GPU probe
139104577 remains separate and untouched. This new path is not GPU-accepted yet.

A study-level finding may occupy only a few windows. Sampling12 of94 can omit
that evidence during an update. Instead of reducing pixels, shrinking the image
or freezing the encoder, the new mixin checkpoint-recomputes each8-window encoder
chunk while the MIL head attends over ALL94original windows together. This is
ordinary nonreentrant activation checkpointing applied to whole-study MIL, not a
claim of a novel ML theorem. It trades extra encoder compute for bounded memory.
The entire encoder and head still receive the complete bag's gradients.

The source-supplied original Raptor class is subclassed privately; no model source
global or live job changes. Its full original state is strictly loaded and all
tensors checked equal before learning. All parent BN running buffers stay frozen
to avoid double updates during recomputation. Dropout RNG/autocast are preserved
by checkpoint replay. Inference/no-grad uses the unchanged parent encoder path,
including its original chunking, and all94two-view windows.

Three actual synthetic CPU tests show exact forward logits, every parameter
gradient, BN buffers and post-backward RNG versus the same8-window-chunk full94
graph without checkpointing, plus rejection of mutable BN and unchanged inference
delegation. Compiler supports a separately bound whole94source. These are
engineering results only; no MRI accuracy or real T4 memory/speed acceptance.

If this path passes its own actual16fitting+16disjoint engineering MRI twoT4
full/stop1/resume recovery and measured runtime, prefer it for the first full
adaptation: same3epochs/3887/1458updates/lowLR/.25source+.75teacher recipe, but
ALL94windows instead of12sampled windows. This preference follows information
coverage and bounded-memory engineering, not tiny diagnostic scores. Full fit
still requires all4cache terminals and full original pixel/label/source/funding
bindings. The12-window diagnostic cannot accept a different encode path.

No whole94GPU/full fit is dispatched by this preparation snapshot. Protected
Gold/confirmation and original accepted labels/folds remain unchanged. No new
competition score is inherited from the released.949initialization.

Actual whole94recovery is now pushed ONCE/RUNNING:139105348v1,
godlydonuts/rsna-parent-whole94-recovery-a45d2a00322d, source60eb779b,
exact notebooka45d2a00322d. CPU939196 admitted original32pixel/source/label joins,
full unchanged initial model tensor equality and the separate checkpointed source.
CPU939197 performed the sole SaveKernel; fresh conservative83131.984GPU seconds.
Original12GPU recovery had actually completed before this new GPU was dispatched.
At most one KaggleGPU VM plus the separately queued own H100job. Onehour max,
no competition entry. Sole finite owner exec70752/PID75947 observes terminal,
never auto-retries/pushes or starts a production fit. No actual whole94acceptance
or accuracy is claimed until its exact terminal receipt.
