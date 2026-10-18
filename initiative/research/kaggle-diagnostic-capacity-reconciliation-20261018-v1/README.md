# Restore shared GPU registration after a partial SaveKernel rejection

Kaggle rejected the finding model's first public run with the explicit maximum
of two batch GPU sessions. Its existing32-study two-T4 qualification passed.
The new diagnostic's namespace was missing from frozen legacy capacity scanners;
this allowed an optimistic request and retained the canonical claim after the
non-success response. The finding and downstream polling controllers stopped;
training notebooks and both original H100 jobs were not restarted.

The response error alone did not establish absence: actual Kaggle UI showed an
owned draft, zero saved versions, Starting Fresh version history and draft
session off. Original profile search correctly refused an ambiguous absence
claim. CPU957964 reconciled this draft-only partial mutation and original failed
CPU957815, preserved the exact original request/source/response, registered the
ALREADY accepted diagnostic139417155 in a distinctly marked capacity-only alias
recognized by frozen scanners, and released only that dead owner's claim. The
alias is explicitly not a serving model. Exact GPU identities deduplicate across
old/new scanner namespaces. No GPU notebook or competition entry was created
by reconciliation.

Original LR E2 and diagnostic status owners were resumed, preserving all original
versions and source hashes; pre-ledger claim contention is a wait, not an unknown
mutation retry. LR final scoring again follows the actual replacement process.
Finding v4 reuses only the original owned zero-version draft after a fresh
GetKernel state check; it refuses any saved version, preserves all prior intents,
uses a distinct next-attempt intent and waits fresh complete capacity/budget.
A future explicit batch rejection releases its claim instead of blocking research.
No first public version has been created by this control admission.
