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
