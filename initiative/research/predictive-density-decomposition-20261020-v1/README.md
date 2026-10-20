# Matched fitting-only density decomposition

Prepared only. No CPU execution, new checkpoint, training, selection or score.

This one-CPU diagnostic compares the original C24 checkpoint and the **actually
completed** C10024 pilot on the same original fitting ordinals0/1, both contexts.
It uses the exact previously executed CPU981494 loader and its input/checkpoint
receipts. The C100 branch additionally requires the original981620 terminal,
accepted recovery, frozen e11c binding, runtime identity, unchanged numerical
modules, exact shared input index/targets and retained final-checkpoint stat.
Pending or failed pilot evidence cannot satisfy preparation.

For each active label/context it separates the original objective exactly:

`A = -logp0/(64*N)`, `D = -q*logBF/(64*N)`, `density = A+D`.

The output reports the scalar and vector-gradient identities, unweighted
component norms, alignment with exam gradients and each other, and the signed
label margin `(2*q-1)*logBF` both before and after normalization. Multiply the
reported component norms by the checkpoint's declared coefficient when comparing
weighted contributions; this does not predict Adam updates.

The same two model forwards also support the existing fixed-head regional-mean
intervention and context-only comparison. The intervention retains case means
and all context-derived predictions. It does not remove all MRI information.
Each checkpoint sees exactly the two original contexts per fitting study.
There are no extra encoder passes for the intervention, optimizer steps,
parameter-gradient writes, or changes to retained checkpoint files. Original
in-memory model tensors are restored even if the comparison fails.

`prepare.py:prepare(source_git, c100_reference)` returns a source/payload capsule
for the existing stdin transport. It never calls SSH or the scheduler. The root
owner supplies a distinct one-time execution journal and explicitly admits a
one-CPU, zero-GPU,12GiB,30-minute invocation on evc27 after the actual pilot
finishes. New remote/shared files: zero.

The exact `c100_reference` fields are:

```json
{
  "acceptance_sha256": "<actual GPU-accepted.json SHA>",
  "checkpoint_stat": ["inode", "bytes", "mtime_ns", "ctime_ns"],
  "terminal_sha256": "<actual full/terminal.json SHA>",
  "scheduler": {
    "GPU_job": "981620",
    "state": "COMPLETED",
    "exit_code": "0:0",
    "node": "evc27"
  }
}
```

The four stat entries must be actual positive integers. Source and metadata
remain local/private; only aggregate results may be published.

A decrease in common reconstruction alone is not evidence of diagnosis
discrimination. Inspect signed density margin, exam-versus-context BCE and
regional sensitivity together. These are two weakly labelled fitting cases at
an early checkpoint, not clinical ground truth or held-out performance.
There is no automatic promotion, coefficient/covariance search, access to the
four engineering cases, or modification of healthy B/C/C0/C100 work.
