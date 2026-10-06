# Label-source disagreement census (2026-10-06)

Join of `llm_labels_v4_blend` (public) against v2 per-cell states across all
4,407 studies × 12 targets. Artifacts: `disagreement_cells.parquet` (14,206
flagged cells with v2 evidence + report_group for targeted re-review) and
`disagreement_report.json` (full matrices).

## Gold truth on the public table

Binary accuracy of `llm_labels_v4_blend` vs the 58 expert labels (threshold
0.5): **mean ≈ 0.80**, range 0.69 (Synovitis) – 0.90 (Baker's). Independent of
the ablation, this quantifies the ceiling any model trained on public labels
inherits — consistent with the ~0.83 extractor-agreement estimate in the
rank-18 write-up. Per-target values are in `disagreement_report.json`.

## v2 positives are corroborated; v2 negatives are the contested cells

Public median value by v2 state, per target (full table in the JSON report):

| v2 state | public median (typical) | reading |
|---|---|---|
| positive | 0.96–0.995 across all 12 | public corroborates v2 positives almost perfectly — v2 positives are reliable |
| negative | 0.025 for most targets, **0.89 Effusion**, 0.91 PF OA (p90), 0.78 Medial OA (p90) | for Effusion/OA the public extractor calls ~half of v2's negatives positive — one side is systematically wrong on mild/gradable findings |
| uncertain (v2 masked) | bimodal, often ≈0.93–0.96 positive | v2 masked cells split into decisive public-positive and public-negative clusters |
| not_mentioned | flat 0.25 (public's silence prior) | consistent — public emits ~0.25 when the report never names the finding |

## The flag taxonomy (counts)

- `uncertain_resolvable_by_public`: **8,167** cells — v2 left masked, public
  decisive (<0.1 or >0.9). Highest-volume review target: adjudicating these
  unlocks thousands of supervised cells for targets where v2 is thinnest
  (PF OA 1,694; Medial OA 1,313; Effusion 982; Contusion 857; Baker's 634).
- `public_over_calls_negative`: **3,082** — v2 negative, public >0.5. Half
  are Effusion (1,384) and Baker's (461); mostly mild-finding threshold
  differences. Verify on gold: public accuracy on Effusion is only 0.724, so
  these are more likely public over-calls than v2 misses — but a v2.1 pass on
  the subset with public >0.9 would be cheap and decisive.
- `uncertain_hedged_both`: 2,922 — v2 uncertain and public hedged; leave
  masked.
- `public_under_calls_positive`: **27** — v2 positive, public <0.5. Tiny, but
  every one is a hard disagreement worth manual re-review (they're either v2
  errors or uniquely identified v2-only positives).
- `public_positive_v2_silent`: **8** — public >0.9 where v2 found no mention.
  Near zero; consistent with the public extractor's low silence prior.

## Prioritized v2.1 review queue (`review_queue.parquet`)

The same 14,206 cells ordered for delta review. Dedup by `report_group` cuts
effective volume ~3.7× — 3,870 unique reports cover all flagged cells, and a
reviewed group resolves every cell it contains.

| tier | cells | content | action |
|---|---|---|---|
| 0 | 35 | `public_under_calls_positive` + `public_positive_v2_silent` | review **all** — possible v2 errors either direction |
| 1 | 1,727 | `public_over_calls_negative`, public >0.9 | decisive conflicts; high info per review |
| 2 | 3,241 | `uncertain_resolvable_by_public`, public >0.95 | likely true positives v2 left masked |
| 3 | 2,321 | `uncertain_resolvable_by_public`, public <0.05 | likely true negatives v2 left masked |
| 4 | 1,355 | remaining `public_over_calls_negative` | mild-finding threshold disputes |
| 5 | 2,605 | remaining `uncertain_resolvable_by_public` | decisive-but-not-extreme masked cells |
| 6 | 2,922 | `uncertain_hedged_both` | leave masked — lowest value |

Within a tier, cells sort by `|public − 0.5|` descending. A v2.1 pass can
stop at any tier boundary and still be a strict improvement over the v2
table; tiers 0–3 alone cover the most contested ~51% of flagged cells.

sha256 `e55c7218e84d1270664466941657a54963d00388022fc24b1dffe35b691751bc`.

## How this feeds the program

1. The ablation (paired fold-0 fit) tests the *aggregate* effect of the two
   tables. This census explains *where* they differ, so the result is
   interpretable either way it lands.
2. `disagreement_cells.parquet` is the input queue for a **v2.1 delta
   review**: only ~14k contested cells (~3% of all decisions), concentrated
   by flag — a delta review is ~50× cheaper than re-reviewing 52,884.
3. Effusion and the OA targets are where the sources diverge most; if the
   ablation favors v2, these flags are the mechanism; if it favors public,
   these are the cells v2 got wrong.
