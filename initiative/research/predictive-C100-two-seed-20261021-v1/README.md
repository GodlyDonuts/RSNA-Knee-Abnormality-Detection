# Fixed ensemble of the independently trained C100 models

October 21, 2026, 19:31 UTC. **Implemented and locally tested; no trained combined GPU
execution or competition entry.** Newton's actual final source is
`055a8f8ed7e0d64f47537fb25480a0f08c7ac947ec8f0472e22f7dcdc117d6e9` and model
`e606d974f4aa8bcc2a4e07375353e41f1e4d31d7151f7d223380624c8116d8d6`.
The independent Colab seed20261021 finished six epochs/2,916 updates, but its
original private preservation and standalone qualification are still underway.
Its actual final deployment hashes must come from that original owner; none
are invented or copied from Newton.

## Hypothesis and fixed decision

Independent initialization and training order can produce different errors.
The candidate averages the two complete models' probabilities **50:50** for
each of the12 targets. This is an ensemble of our predictive-evidence models,
not a new architecture or evidence of a breakthrough. Both trained on the same
3,887 fitting studies and inherited encoder, so correlated errors could erase
any benefit. No label, public-score weight sweep, per-target tuning, rank
transform, or parameter averaging is introduced.

Use an actual combined test only after both original standalone hidden runs
have scored without errors and each reaches at least0.949. A qualifying score
does not guarantee an ensemble gain. Protect the original Colab deployment
and final selections before spending further GPU time or a competition slot.
The last observed freeGPU2423.82seconds minus Colab1920 leaves503.82seconds,
less than this candidate's900-second public cap plus120-second startup reserve.
There is no funded combined GPU dispatch at this snapshot.

**Actual outcome now blocks this candidate:** original Newton submission57491555
returned **0.692**, COMPLETE with381,996 output bytes and no reported error.
It fails the fixed0.949 criterion. Preserve this implementation, but do not
compile/run the trained ensemble or use a slot on it. Colab's standalone
comparison remains separately authorized.
[Original scored receipt](../predictive-C100-final-entry-20261021-v1/actual-scored.json).

## Implementation

`compile.py` takes the exact original standalone sources, actual trained T4
qualification/public receipts and their original private public-example CSVs.
It binds fixed-final epoch6/update2916,24-context models, complete candidate
identities, runtime, original Dataset and serving hashes. Actual Colab
preservation/compilation remains owned by consumer8603; do not create a second
Colab export or restart that consumer.

`supervisor.py` runs the unchanged Newton and Colab sources sequentially in
fresh processes. Each child owns both T4s while active. After each child exits,
the original working directory, outputCSV and acceptance are moved intact to
distinct private directories before the next child reuses the same paths.
One shared nine-hour watchdog covers both models; the exact public roster has
a900-second cap. Failure cannot leave a partial finalCSV or invoke a fallback.
The first local timeout test exposed a race between timer and main-thread
process-group cleanup; serialization fixed it, and actual subprocess timeout
and descendant-cleanup tests subsequently passed.

The joint public run must compare each child's36 probabilities against its
own original public bytes, with maximum absolute difference<=1e-4. References
are used solely for that public engineering check, never for hidden prediction
or as a fallback. All hidden rows are recomputed from MRI by both models.
`fusion.py` validates exact13-column schema, full official roster, unique rows
and finite[0,1] values, then writes their equal mean in official order.

The private compiled source contains original external source and public study
identifiers; keep it on authorized private cluster/Kaggle storage, off Mac/Git.
No new modelDataset, model copy, training job or publication is required.
No dispatch/entry coordinator for this branch has been executed or added yet.

## Evidence and next action

Fifteen focused tests passed, including real isolated subprocesses, shared
timeout, descendant cleanup, original-path archival, failure cleanup,
UID-aligned probability averaging and exact source/model/receipt admission.
These are software checks, not actual GPU or accuracy evidence.
[Exact source hashes and local acceptance](acceptance.json).

After original Colab's seven private proofs and standalone public acceptance,
collect its existing source/receipts on the owning node, compile once, and
admit fresh storage/account capacity. After both real standalone scores meet
the fixed criterion, run the exact combined public candidate once. Assess its
measured total runtime against the nine-hour hidden limit before an entry.
Reconcile any uncertain mutation read-only. Preserve R6's actual0.958 and
settle the final two selections before October22 23:59UTC; target any new
scored launch byOctober22 noonUTC.
