# Fixed-final rank64 fold-0 result

Newton job **865578 completed 0:0** on evc31 in 6:47:01. Exact result.json,
run.json and 1,156-study predictions are preserved here. The ten-epoch run
used 2,247 training studies and 2,810 optimizer updates. All ten immutable
checkpoints were privately published and fully downloaded/size/SHA verified;
E10 revision is `4b81ebc1b9fd099287247cc72061be1a620d02a2`.

Independently recomputed fixed-final internal macro AUC is
**0.8340496537270812**, versus **0.8620940794159951** for sampled rank32 on
the same held-out studies and masks: difference **−0.028044425688913965**.
These are report-derived development metrics, not Kaggle scores. Neither
gold nor reserved confirmation was used.

`recompute.py` validates prediction bytes, run/result bindings, accepted label
bytes, fold membership, train/outer UID hashes and all twelve target AUCs.
Run from the repository root:

```bash
python -c 'import runpy; runpy.run_path("initiative/audits/dinov3-rank64-fold0-result-v1/recompute.py", run_name="__main__")'
```

comparison.json preserves the original replay; comparison-v2.json adds label
hashing and recipe differences. Both retain all twelve targets. MCL has only
**two positive studies in two components**, against 946 negatives. Its AUC
fell from 0.6569767442 to 0.3477801268; this contributes about 92% of the
total macro decline. Leaving one positive component out gives wide ranges
for both models. This sensitivity does not justify dropping MCL or claiming
higher rank universally harms performance. ACL and Synovitis also declined;
Effusion and Lateral OA rose.

Rank64 has 20,447,232 adapter parameters versus rank32's 10,223,616. Changing
adapter shape can change RNG consumption and head initialization. Delta storage
and explicit configuration defaults also differ, as recorded in recipe_changes.
This is not a causal rank-only experiment. Rank32 remains the preferred measured
recipe; whole-series and original-source native384 experiments test coverage
and detail separately. No rank64 competition submission was made.
