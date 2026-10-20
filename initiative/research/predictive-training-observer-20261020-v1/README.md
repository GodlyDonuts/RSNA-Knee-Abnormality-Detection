# Read-only production training observer

Prepared locally; not launched. This independent finite observer uses portable
the exact new `predictive-v4-owner-handoff-20261020-v1` owner intent and its immutable source manifest, preserving portable base `926b6577` and numerical source `44cc559e`. It shares the successor launcher’s pure read-only capsule builder; it never calls its freeze/launch path. Dispatch receipts come only from the new successor journal/audit directory.
It waits for the sole fitting owner's actual B/C-on-evc33 and C0-on-evc27 dispatch
receipts, then invokes only the existing `mode=progress` via Newton's management
Python. It reads the original shared logs and scheduler state; it allocates no
CPU/GPU, reads no credentials/pixels/checkpoints and performs no remote mutation.

The unchanged worker already records finite losses, pre-clipping gradient norms,
learning rates, synchronized update timing, cumulative context load, peak
allocated GPU memory and peak host RSS. Update146 reaches the planned `3e-5`
learning rate; the first200 therefore covers the previously unexercised peak.
This observer retains the original progress function's first200 summary once per
arm and atomically refreshes sanitized latest telemetry locally. It does not
parse raw logs independently. There are no retained logBF/logit distributions,
so these observations establish neither saturation nor clinical accuracy.

Read-only SSH failures retry for at most30minutes. A validation/nonfinite
rejection is preserved as failure evidence and stops the observer; it never
cancels or restarts the GPU jobs. Missing dispatch receipts cause waiting, not
dispatch. The observer ends when all three jobs are terminal or at
October22,09:00UTC. Its single-start journal is
`artifacts/predictive-training-observer-20261020-v1/`.

After review and commit, the root may run:

```sh
python3 -B initiative/research/predictive-training-observer-20261020-v1/observe.py \
  --source-git <exact-committed-observer-source>
```

Outputs `actual-first200-{B,C,C0}.json`, `actual-latest.json` and any
`actual-observation-failure.json` contain only aggregate telemetry and original
job/source/receipt bindings. They are evidence only after actual collection.
