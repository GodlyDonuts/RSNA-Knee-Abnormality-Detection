# Preserve the efficiency option without changing the accuracy goal

14 October 2026. Rules rechecked against the official competition overview. No submission, selection or agreement change performed.

## Verified rule

The competition has an efficiency track with $7,000, $6,000 and $5,000 prizes. An eligible entry must be selected for the main leaderboard award (including applicable automatic selection) and beat the sample-submission benchmark on private data. One submission may qualify for both tracks. The minimized metric is:

    E = A / (B - Amax) + T / 32400

A is its private macro-AUC, B the benchmark score, Amax the highest private AUC, and T evaluation runtime in seconds. Public efficiency ranks are displayed during training; final efficiency evaluation uses private data. [Official competition overview and efficiency evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/)

## Consequence derived from the formula

Let D=Amax-B>0. Candidate C beats incumbent I on efficiency exactly when:

    (A_C-A_I)/D > (T_C-T_I)/32400.

For a hypothetical D=.45, gaining .01 AUC compensates for 720 seconds of additional runtime. Losing .01 AUC would require saving more than 720 seconds. This is algebra under a declared assumption, not a forecast of the unknown private maximum or either candidate's score. The exchange rate changes with D.

Preserve the accuracy/runtime Pareto frontier among already evaluated eligible single models: a candidate worse in AUC and slower is dominated; a slightly less accurate but materially faster one may remain useful. Public ranks and point scores do not resolve private uncertainty. A very fast .95 model could dominate a slower .95 model, so no prize probability can be inferred from a runtime target alone.

## Practical recommendation

Keep the user's main objective of the strongest single MRI model. Record actual complete notebook runtime beside each already-authorized candidate's score, and retain the qualified Pareto candidates until the final selection decision. This requires no extra model family, no extra fit, no automatic final selection and no new protected evaluation. It adds a reason for the compact candidate's direct-deployment option rather than treating it only as a teacher.

Do not equate encoder GMACs, a three-study notebook, submission-to-score latency, or a cited competitor runtime with our full execution time. One-time setup, decoding, loading and output costs matter; use the actual official receipt when available. No current efficiency position or winning runtime has been established here.
