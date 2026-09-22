# Official DINOv3 weight compatibility

The exact approved DINOv3-L checkpoint predates the layer namespace used by Transformers 5.6.1. Its built-in `from_pretrained` path provides a key conversion, but our offline strict loader bypassed that path. Actual GPU pilot 838580 rejected all 408 legacy layer keys before training; its original failure remains in the deployment archive.

The adapter now admits the reviewed architecture, source, revision, config SHA, weight SHA, file format and library version before renaming only leading `layer.` to `model.layer.`. It preserves original file/tensor values, rejects mixed namespaces and still checks every key and shape strictly. Trained-checkpoint provenance records this conversion. Canonical checkpoint-only reconstruction retains its strict full-state validation.

Newton CPU job 838689 verified all 415 official tensors after the 408-name conversion, left seven names unchanged, ran a finite 224-pixel forward with the full 303,129,600-parameter encoder and reproduced its output exactly through checkpoint-only reconstruction. The model process used about 3.18 GiB peak RSS. It used no GPU, medical image or download, and does not certify CUDA recovery, epoch speed or accuracy.

All 905 repository tests, Ruff and mypy passed against the recorded source hashes. The 110 focused author tests and independently checked actual CPU proof are preserved. The next gate is deployment of this exact committed adapter and the bounded two-H100 recovery pilot.
