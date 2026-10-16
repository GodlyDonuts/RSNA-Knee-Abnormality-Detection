# Final context head against the observed cross-run CNN drift

The regional diagnostic discovered cross-run global-feature differences even with
bit-equal observer/no-observer forwards. Rather than infer that they are harmless,
CPU936870 applied the **same actual final context component59ca63ca** to both
original and newly captured global features/parent logits on the exact32MRIcases.
No CNN, new GPU, fitting, clinical labels/reports or protected values were used.

Maximum adjusted mean-TTA probability gap **0.0001292825**, mean absolute gap
0.0000110987; original parent mean-TTA gap0.0001237988. This passes the existing
predeclared0.003T4probability tolerance. The final head did not materially amplify
this observed drift on these32cases. It does not identify the numerical cause,
prove a full-population bound, establish disease accuracy or constitute a score.
No checkpoint, scheduled entry, accepted labels or numerical source was changed.
Private feature bytes and case-level differences stayed on evc33.
