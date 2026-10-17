# Actual S6 classifier and cross-lineage fitting-only screen

Observed October17 06:25UTC. CPU942806 **COMPLETED0:0**, scoring all eight original
fitting MRIs/5,372 windows in1,197.87seconds, peak6.62GBRSS, finite8×1×12 logits,
zero fallback. This is CPU FP32, not GPU timing or clinical validation. The
unchanged original model/feature/pooling code has the same four device/report
AST replacements as the accepted public CPU reference. Exact source6e87350d,
model0c7297d5 and immutable private input139165316v1 are bound in
[the actual result](classifier/accepted.json). Do not repeat this classifier.
The inherited statistic named `gpu_s` describes CPU execution in this run.

CPU942957 then actually reused the existing CoAtNet teacher arrays and those
S6 logits. It decoded **only fitting IDs and model outputs**, without labels,
masks, reports, Gold, new CNN inference, fitting or a GPU. Both complete original
input/output hashes and exact eight-study membership were verified. CoAtNet is
the mean of sigmoid probabilities over its two anatomical views; S6 uses its
original sigmoid output. Case-level values remain private on node33.
[All-target aggregate result](comparison/accepted.json),
[nine numerical/guard contracts](comparison/checked.json).

| Target | Mean absolute probability gap | Rank correlation | Strict pair reversals/28 |
|---|---:|---:|---:|
| ACL | 18.30pp | 0.9524 | 2 |
| MCL | 32.33pp | 0.9286 | 3 |
| Medial Meniscus | 12.61pp | 0.9524 | 2 |
| Lateral Meniscus | 24.36pp | 0.9762 | 1 |
| Medial OA | 12.47pp | 0.8095 | 5 |
| Lateral OA | 21.28pp | 0.9524 | 2 |
| PF OA | 8.04pp | 0.8571 | 3 |
| Effusion | 10.66pp | 0.8095 | 5 |
| Synovitis | 21.12pp | 0.9286 | 2 |
| Baker's | 16.00pp | 0.9524 | 2 |
| Contusion | 17.51pp | 0.9524 | 2 |
| Fracture | 39.09pp | 0.9048 | 3 |

## Decision and limits

Large raw probability gaps coexist with substantially similar ordering. We
therefore **do not use the raw gap as a clinical disagreement/confidence gate**.
Different calibration or priors are plausible explanations, not established
causes. There are no strong opposite cases under the predeclared0.8/0.2 rule.
Eight deterministic series-count quantiles are neither random prevalence data
nor independent OOF; both released encoders exposed this cohort. These values
cannot select disease labels, per-target blend weights or claim an AUC gain.

The practical next comparison is fixed equal-rank fusion and an ordinal teacher
transport into a fully trainable single CoAtNet. The
[new transport mechanism](../../experiments/competitive-v1/s6-rank-transport-v1/README.md)
is numerical preparation, not an admitted trainer. A
[32-study source scout](../s6-fitting32-scout-preparation-20261017-v1/README.md)
checks actual T4 input/CPU parity and throughput before funding full3887 S6
teacher acquisition. It has a sole finite owner and zero competition entries.
Original S6 offline/public acceptance and its distinct entry remain owned
separately; this screen does not replace their acceptance or hidden outcome.

All5372 original fixture tensors, compressed input bytes, teacher arrays and
case predictions remain on authorized private storage. Only the sanitized
accepted receipts, numerical helper, wrapper and original receipt hashes are
archived. No new data transfer, label correction or clinical inference claim.
