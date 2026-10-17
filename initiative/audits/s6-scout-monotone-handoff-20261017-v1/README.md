# Preserve actual public acceptance across S6 submission states

Observed October17 06:38UTC. Source inspection found a scheduling race: original
S6 entry20994 advances the **same canonical record** from public-accepted into
submitted/scored. The scout's single-state admission could then wait forever
despite an intact accepted offline notebook. No actual model or entry failed.

We paused only CPU observer24413, let its read-only phase finish, and required
no live child. CPU943176 then actually confirmed **no scout push intent/response,
no collection and no prepared scout record**, plus unchanged staged GPU source
SHA0573a2a1. Only after that proof did old24413 terminate. The first login-only
read looked for node-local source on the login host and failed FileNotFoundError;
it made no writes. The owning-node read corrected observation, not a job rerun.
[Actual owning-node reconciliation](observer-handoff-verified.json),
[terminal CPU handoff](observer-handoff-complete.json).

New sole **PID26454/exec91663**, local
`artifacts/s6-fitting32-scout-preparation-20261017-v4/finish_once.py`, uses the
same scientific source, staged root, slug, canonical key, data, budget and
maximum one1,800second GPU session/one private collection/zero entries. It now
requires the **identical actual offline public receipt** through accepted,
submitted, scored and hidden-failed states. It verifies exact source/model,
version1,9h setting, twoT4s,3MRI/13columns, zero fallback and B-only weights.
An uncertain-entry state or qualification-only state does not authorize a GPU.
Fresh total future VM funding and actual capacity gates are unchanged.

Fourteen actual pure guard tests cover every allowed transition and pinned
source/runtime/model/receipt refusal. There is still one SaveKernel call site
and no competition-entry API. The original five-path finite transaction AST is
identical; only prelaunch terminal/handoff proof and acceptance-state admission
changed. An initial local static check searched the wrong API-call spelling;
it stopped before launch, was corrected to inspect the actual `r.call` AST and
is preserved as preparation-only. No failed or uncertain push was repeated.
[Guard tests](check_handoff.py.txt), [actual result](monotone-acceptance-checked.json),
[new canonical phase](run_phase_once.py.txt),
[finite-owner preservation](finite-owner-preservation.json),
[new sole owner intent](finish.intent.json).

Never restart old24413, repeat the original stager or create a second scout.
Original S6 reproduction20978/entry20994, main fitting96833, sampling97530 and
expanded GPU938284/watcher95408 were untouched. The staged32-study check still
has no GPU result at this handoff; there is no clinical or0.97 accuracy claim.
