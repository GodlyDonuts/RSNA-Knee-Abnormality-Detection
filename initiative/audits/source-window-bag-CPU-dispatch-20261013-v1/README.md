# Actual CPU-only common initialization admission dispatch

Source971d7dac after169 focused and3138 full passing tests; one actual one-CPU
zero-GPU step dispatched once on evc31, existing parent899766. No new GPU node
or fitting has been reserved. Sixteen-MiB source/receipt control budget admitted
under fresh free-space proof, retaining all prior obligations and20GiB reserve.
No checkpoint budget is granted by this CPU admission. Source/core dataset and
label-loading bytes must remain identical to the old input proof; only reviewed
trainer dispatch and window-bag module differ. Actual common official
mean/query parameter values/RNG/frozen base still await the original terminal.
Do not duplicate dispatch or infer acceptance from a prepared worker.
