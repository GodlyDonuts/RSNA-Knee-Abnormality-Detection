# Matched fitting-only density decomposition

**Actual C24 measurement completed:** CPU 982819 evaluated the original update 24
checkpoint on the same two fitting cases and two contexts per case. It took
69.40 seconds on one CPU, used 2.849 GB peak RSS, wrote no remote files and performed
no optimizer steps. The checkpoint state/stat remained unchanged.
[Original aggregate receipt](actual-C24-result.json). The matched C10024
comparison remains prepared and requires actual pilot completion; it has not run.

## Actual objective balance at C24

The density objective splits into common reconstruction `A` and label-weighted
evidence `D`, as defined below. The following are L2 norms across all nonencoder
parameters before clipping or Adam; percentage ratios apply the actual
`lambda=0.1` once.

| Quantity | Fitting ordinal 0 | Fitting ordinal 1 |
| --- | ---: | ---: |
| Exam gradient norm | 7.08890 | 4.83777 |
| Unweighted common `A` gradient norm | 0.00725464 | 0.00677389 |
| Unweighted label-weighted `D` gradient norm | 0.000317360 | 0.000145938 |
| `lambda * ||grad A|| / ||grad exam||` | 0.010234% | 0.014002% |
| `lambda * ||grad D|| / ||grad exam||` | 0.000448% | 0.000302% |
| Cosine of `grad D` with exam gradient | 0.5842 | 0.3752 |

Common reconstruction has 22.9–46.4 times the gradient norm of the label-weighted
term. Its cosine with the exam gradient is only 0.0084–0.0113; `D` is smaller but
more aligned with the exam objective on these cases. In the shift head,
`grad A` is exactly zero, and the weighted `D` norm is 0.000404–0.000540% of the
exam norm. The scalar decomposition agrees within 5.96e-8 and the aggregate
gradient sum within 1.22e-7 relative L2 error. The Gaussian constant has zero
gradient; its large scalar contribution does not explain gradient strength.

Signed supported-versus-wrong diagnosis margins are positive but small:
0.00533–0.00791 before division by the 64-coordinate/observed-region denominator,
and 1.55e-7–1.64e-7 afterward. Full-posterior minus context-only BCE is
**+0.001033** on one case and **−0.000951** on the other. Replacing regional
descriptors with within-case coordinate means changes probabilities by only
0.0205/0.00506 percentage points, versus 2.516/2.049 points when removing the entire
Bayes-factor term. These observations support testing objective balance and
descriptor geometry separately. Raising the density coefficient scales both
`A` and `D`; it does not selectively strengthen diagnosis discrimination.

Cached-image summaries also show low nonfill intensity: after excluding exact
0.5 fill, means are 0.0518–0.0564, medians 0.00841–0.0492 and standard
deviations 0.0656–0.0811; exact fill occupies 57.5–60.0% of elements.
This is **not an anatomical visibility mask**: background and interpolation
mixing remain, and legitimate values exactly 0.5 are also excluded. These
summaries cannot establish a preprocessing defect, visibility failure or the
cause of weak regional sensitivity.

This is CPU-FP32 evaluation with detached encoder features on two weakly
labelled fitting cases at update 24. It does not measure encoder gradients,
clipped/Adam updates, late-training behavior, held-out accuracy or C100 effects.
No live B/C/C0/C100 recipe, variance, descriptor normalization or model selection
changed as a result of this measurement.

## Execution contract

`prepare.py:prepare_baseline(source_git)` separately prepares an explicitly
**C24-only** measurement now. It needs no C100 reference, never opens the C100
root, and never loads a second model. Its result status is
`actual_C24_only_fitting_density_decomposition`, with exactly `C24` entries.
Passing C100 references to that execution mode fails. The original C24
checkpoint/input proofs and exact original runtime remain mandatory.

The separately prepared paired diagnostic compares the original C24 checkpoint
and the **actually completed** C10024 pilot on the same original fitting
ordinals0/1, both contexts.
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
