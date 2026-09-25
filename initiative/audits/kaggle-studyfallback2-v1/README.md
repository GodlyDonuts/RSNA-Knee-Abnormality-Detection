# Fold-2 epoch-8 deployment correction

This is a distinct Kaggle code-notebook candidate using the already trained
fold-2 epoch-8 global-CLS DINOv3-L checkpoint, SHA-256
`eb280864bd1c2f896210ce360e1ae36d59649c854a9051523c2454a34d08c7b1`.
The exact model runtime is pinned to training Git
`b74060b98f01bff1b924070ccba5ce44dd002f40`; the packaging and inference
wrapper are projected from Git
`94215dce22a8ec8bbf4224037f8488bc4f23ebd0`. The projection manifest
SHA-256 is `0f2adcf19e46bb91478d27993cdbb3e402cb87d339252a9523434a5613e3c942`.
The [projection receipt](projection-receipt.json) and exact Newton launch
scripts are archived here.

The earlier code submission **56528793** used the same checkpoint with a
different wrapper and notebook. Kaggle marked its hidden rerun complete with
zero output bytes, an unhandled-notebook error and no score. The hidden
traceback was not exposed. The new wrapper tolerates a bounded class of
unusable MRI studies and malformed DICOM attributes; tests prove those cases
but do not prove that either caused Kaggle's hidden failure. A scored result
is not claimed until Kaggle supplies one.

The initial candidate mistakenly selected epoch 9. Slurm preparation job
848845 stopped in the eligibility guard before creating a Dataset, notebook
or competition submission: only the immutable scheduled even epochs are
eligible. That stopped attempt is preserved as an audit, never retried.
Eligible epoch-8 preparation was then dispatched once as Slurm job 848860,
to the distinct private bundle
`/home/sa305415/rsna-knee/kaggle/bundles/fold2-epoch008-eb280864bd1c-studyfallback2-v1`.
The prospective Dataset and notebook slugs are
`godlydonuts/rsna-knee-f2e008-studyfallback2-eb280864bd1c` and
`godlydonuts/rsna-knee-epoch8-f2-studyfallback2-eb280864bd1c`.
All publication and submission operations must go through the single Newton
account ledger and one-CPU coordinator. The old failed operation is archived
only after fresh authenticated evidence confirms its exact terminal failure.

Preparation job 848860 completed `0:0`. The distinct private Dataset was
created once as ID **12183673**, version 1, and every payload object was
downloaded and SHA-256 verified; see `dataset-verified.json`. The exact private
offline T4 notebook version 1 was pushed once. Its source SHA-256
`56ecd1ee81ce09fe6756f9cb5640c594b8c1d3eda4e3b95eee960e226469ae36`
matched the prepared operation. Kaggle completed its public MRI-only run;
`public-example-receipt.json` and `public-example-only.csv` prove three official
IDs in order, 13 columns, finite probabilities in `[0,1]`, and CSV SHA-256
`a3d29bc7f8c6908b6db13162bb495e76623f837c3caa4d3bb40067250acf0350`.
That three-row CSV was **not** uploaded to the competition.

After a fresh live allowance of five, the single coordinator submitted the
exact code notebook once as **56546143** at 2026-09-25 08:50 UTC. Its first
authenticated observation had no hidden status or score and four available
slots. Hold all further submissions until its hidden rerun reaches a real
terminal outcome; public execution does not prove hidden success.
