# Knee predictive-evidence research package

Read `RESEARCH_PLAN.md` first. This is a proposed research direction, not a Kaggle-ready model.

Files:
- `RESEARCH_PLAN.md`: formulation, derivation, data contracts, controls, execution budget, risks, primary sources.
- `predictive_evidence.py`: differentiable reference probability calculation and weak-source loss.
- `test_predictive_evidence.py`: 18 synthetic mathematical tests.
- `test_results.json`: actual CPU test result.

Run with Python and PyTorch:

```bash
python test_predictive_evidence.py
```

The reference requires FP32/FP64 likelihood arithmetic. It deliberately rejects trainable targets, malformed observation masks, nonfinite observed descriptors and unsupported variance values. It does not contain an MRI loader, a trained backbone, registration, protected-cohort data, a Kaggle submission routine, or any external medical weights. Production geometry, input masking, caching, optimizer recovery and runtime qualification remain implementation tasks.

No network calls or repository mutations occur when running these tests. The dense support-mask representation is a correctness oracle for small examples, not the recommended large-study implementation. Clinical results cannot be inferred from the test count.
