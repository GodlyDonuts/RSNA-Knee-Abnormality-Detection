# Full-map spatial reader: actual inputs and bounded real-engine qualification

The new reader consumes the original N3E16 **full1024×12×12 maps**, learns a
1024→64 projection and twelve evidence channels, and computes shared global
statistics plus within-acquisition spatial/longitudinal co-support. Its common
head starts at zero and has no hard correction ceiling. The encoder and all
original image/window selection remain frozen. No cross-plane registration,
lesion localization, new clinical result or Kaggle gain is claimed.

The main control preserves observed tokens and weighted marginals while
**randomizing geometric links**; retaining every positive geometric edge would
leave most alignment intact. The previous exact-support operator stays available
as a targeted distance-weight diagnostic. Both new arms retain common image,
global, nuisance, supervision, initialization, optimizer and stopping paths.
Nominal token coordinates are not local receptive-field or lesion annotations.

Actual CPU950079 admitted all32 original scout studies/2345 windows, with84local
and97repeat groups in fitting and84local/107repeat groups in prediction-only
engineering outer cases. Maximum weighted-marginal error2.7939677e-9 and centered
operator Frobenius ratios0.81917–0.83952 establish a usable control, not clinical
accuracy. The first probe stopped on a stale storage floor before staging; the
second exposed slow IPF convergence. All originals/failures remain preserved.
A geometry-only numerical fix passed33focused tests and the complete real input
plan. CPU950100 independently admitted16original masked binary fitting targets
and16disjoint prediction-only outer cases, without loading outer target values.

[Engineering decision](engineering-decision.json) binds the proposed actual GPU
full/stop1/resume checks for both arms and an online-versus-cached exact engine
comparison. Its source is committed locally before dispatch; GitHub publication
waits for the complete source-frozen test suite. No full reader fitting has yet
been dispatched. The GPU wrapper is diagnostic only and cannot submit a model.

[Budget reconciliation](actual-budget-retirement.json) checked all16 original
N3 publications and terminal bindings without repeating bulk hashing/transfers.
Only8GiB of its unused original24GiB future allowance was retired; all originals
stay in place with16GiB retained. This is not deletion credit. Source/engineering
controls draw from the existing512MiB reader allowance inside the3GiB scout
root. Later full fitting still needs actual GPU acceptance and fresh funding.
