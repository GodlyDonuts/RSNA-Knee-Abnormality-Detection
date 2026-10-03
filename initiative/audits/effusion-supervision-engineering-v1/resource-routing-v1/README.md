# Resource routing: correct an oversized host-memory request

Live Slurm exposed a free H100 on owning evc31, but only23265MiB of
scheduler-unreserved host memory. The64GiB request blocked original869300
until an estimatedOctober10 start. The prior bounded initialization attempt
peaked at9268160KiB; corrected full recovery peak remains unmeasured.
The same still-pending job was updated to22528MiB/22GiB, preserving the
original oneH100/twoCPU/source/config/node/output/runtime. No cancellation,
resubmission, duplicate or device-isolation bypass occurred. It then actually
allocated evc31 and passed driver/NVML/header startup guards. Numerical
recovery and clinical accuracy are not established by this resource change.

An initial22G input was rejected by scontrol before any mutation. Read-only
reconciliation confirmed the unchanged64G request before submitting the
explicit22528MiB update; both attempts remain privately archived.
Full production memory limits are unchanged; production peak RSS is much
higher than this bounded engineering case. Do not reuse22GiB for full fitting.

Authenticated Stokes inspection found no advertised GPU GRES; its CPU
nodes use the same shared Lustre quota. It is suitable for independent
CPU work, not an alternate GPU for this test. Authenticated Kaggle Settings
shows GPU+Colab usage00:00/45hours. No Kaggle job or competition submission
was launched; exact candidate/data/runtime/ledger admission still applies.
No raw browser/account-personal data or other users' job details are published.
