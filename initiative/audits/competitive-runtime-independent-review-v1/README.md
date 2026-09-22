# Independent competitive runtime review

Reviewer/executor: `/root/competition_contract`; runtime implementation owner: `/root`. Exact reviewed file hashes and test scope are in `review.json`.

No remaining blocking correctness defect was identified in the inspected runtime after the fixes below. This is a bounded source review plus synthetic CPU execution, not proof of GPU numerical equivalence, cluster preemption behavior, hidden-test timing or competitive performance.

The review identified two concrete defects and confirmed their corrections:

1. Dataset item seeding used `torch.manual_seed` while preserving only CPU RNG state. With main-process loading, that could reset CUDA randomness. The sampler now seeds only `torch.random.default_generator`, and the root regression forbids the broad seeding call.
2. Competitive inference initially checked only selected model/data code hashes, leaving shared DICOM decoding and preprocessing code unbound. It now requires the complete current `rsna/**/*.py` hash dictionary to equal the checkpoint's training dictionary. Independent tests reject changed model, decoder, preprocessing and shared inference hashes before encoder loading.

The review also raised the risk of other DDP ranks waiting through rank-zero-only full inference. Final inference now uses equal-length rank schedules, drops padding from predictions, coordinates cancellation before each batch, gathers exact unique UIDs, and aligns labels by UID. Independent two-rank Gloo tests cover one real study (one rank has no real predictions), a five-study padded tail, and cancellation requested on only one rank. Those tests passed.

Inspected properties include full release validation before train-only fold selection; exclusion of reserved weak validation/gold; effective-batch per-target loss counts across accumulation and ranks; zero contribution from masked/padded samples; skipping all-masked optimizer updates; fixed final-epoch outer evaluation; exact source/config/code/world-size checkpoint binding; atomic checkpoint writes with the configured disk reserve; per-rank RNG restoration and deterministic resumed sampling; and exact offline test-template UID/column validation.

The reviewer independently ran all 20 root training tests plus seven new adversarial runtime tests: **27 passed**. These include actual synthetic-model optimizer updates, two-rank global-gradient equivalence, and exact CPU mid-epoch resumed checkpoint/prediction equality. Ruff check and format check passed for the independent test file. The system Python lacked pydicom during the first collection attempt; the successful run used the project's existing dependency environment. That environment issue was not a runtime defect.

GPU/BF16/NCCL behavior, actual Slurm signal timing, different GPU hardware across resumed allocations, catastrophic worker failure, and the complete hidden-test inference budget still require allocated-node acceptance. Device type and software/world-size binding do not establish bitwise equality across different CUDA hardware. Runtime checkpoints are trusted local artifacts; these tests do not claim authentication against arbitrary checkpoint replacement. Root's separate actual accepted-data CPU and distributed acceptance remains separately attributable.

The related Newton supervisor review found that a fresh periodic checkpoint alone could misclassify an unrelated failure as a graceful pause. Its owner separately added an explicit fresh trainer stop receipt bound to the run and checkpoint hash. The supervisor is not owned by this reviewer and is not included in the 27-test runtime claim.

The fold/evaluation implementation was authored by this reviewer, so it is not self-certified as independently reviewed. A distinct agent, `/root/competition_contract/medical_encoder_sources`, separately read those files, ran all 30 focused tests, and found no material blockers. Its message-based assessment is separately preserved in this directory with exact file hashes and honest authorship.
