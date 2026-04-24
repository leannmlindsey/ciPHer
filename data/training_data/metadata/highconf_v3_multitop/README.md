# Highconf v3 — multi-top training lists (broad-range aware)

Four protein-ID lists for **per-head independent training**, derived
with a **multi-top** HC filter that rescues broad-range TSPs/RBPs the
v2 strict filter was systematically excluding.

## Why v3 exists

The v2 filter (`highconf_v2/`) requires each cluster's top-1 serotype
to hold ≥ 90 % of deduped supports. Audit A15 (see
`CLAUDE_PHI_DATA_ANALYSIS/analyses/15_broadrange_retention/`) showed
this drops broad-range proteins at ~10 × the rate of specific ones:

| bucket        | v2 K retention | v2 O retention |
|---------------|----------------|-----------------|
| specific (1)  | 43 %           | 28 %            |
| narrow (2)    | 11 %           |  6 %            |
| broad (3-4)   |  5 %           |  3 %            |
| very broad (5+) |  3 %         |  4 %            |

Meanwhile ~20 % of PhageHostLearn and PBIP phages are broad-range, and
UCSD is 73 % broad — so the model has essentially no training exposure
to that biology.

## The multi-top rule

A cluster is "highly confident" if the **top-N serotypes (N ≤ 3)
together** sum to ≥ 90 % of deduped supports. This matches the model
config's `max_k_types: 3` and `max_o_types: 3`. Each protein still
inherits its own labels from its association rows via the existing
`label_strategy: multi_label_threshold` — we just stop pre-filtering
broad-range proteins out of the data entirely.

## The files

| File | Proteins | K-types | O-types | Filter |
|---|---|---|---|---|
| `HC_K_cl95_multitop.list` | 35,249 | 161 / 161 | — | Strict: 95%-cluster top-3 ≥ 90% for K |
| `HC_K_UAT_multitop.list` | 37,158 | 161 / 161 | — | Union across thresholds 60/70/80/85/90/95 |
| `HC_O_cl95_multitop_full_coverage.list` | 35,429 | — | 22 / 22 | cl95 multi-top + O11α,2β rescue |
| `HC_O_UAT_multitop.list` | 37,178 | — | 22 / 22 | UAT multi-top (full coverage without rescue) |

Size vs v2: K grew +51 % / +43 % (cl95 / UAT), O grew +141 % / +139 %.

## Two training recipes

- **Strict (multi-top)**: `HC_K_cl95_multitop` + `HC_O_cl95_multitop_full_coverage`
- **Maximal (multi-top)**: `HC_K_UAT_multitop` + `HC_O_UAT_multitop`

Both are drop-in replacements for the v2 pair. Train both and compare
PHL / PBIP / UCSD rh@1 to decide which wins.

## Retention by specificity bucket (v3)

**K-axis:**

| bucket | HC_K_cl95_multitop | HC_K_UAT_multitop |
|---|---|---|
| specific (1) | 62.8 % | 66.2 % |
| narrow (2) | 43.2 % | 45.9 % |
| broad (3-4) | 24.6 % | 26.2 % |
| very broad (5+) | 9.6 % | 11.6 % |

**O-axis:**

| bucket | HC_O_cl95_mt_fc | HC_O_UAT_multitop |
|---|---|---|
| specific (1) | 62.9 % | 65.7 % |
| narrow (2) | 50.1 % | 53.9 % |
| broad (3-4) | 34.8 % | 39.0 % |
| very broad (5+) | 21.4 % | 24.7 % |

## Under-served serotypes (<5 proteins retained)

All 161 K-types and 22 O-types are present in every list. Very small
serotypes (e.g., `KL175` with 2 pipeline_positive proteins) appear but
won't carry much gradient signal — same caveat as v2.

## Note on the O head specifically

See the A16 entry in `CLAUDE_PHI_DATA_ANALYSIS/lab_notebook.txt` for
the full biological discussion. Short version: TSPs are primarily
K-targeting; some "broad-O" labels in the training data may reflect
host-background variety rather than genuine O-recognition. If O v3
doesn't improve validation-set rh@1 over O v2, that's evidence the
expanded O training set is absorbing host-background noise. Worth
A/B testing O v2 vs O v3 while K sticks with v3.

## Still blocked on

The training pipeline's per-head data-loading refactor (loss-masking
per head, agent1 owns) — same blocker as v2. See
`handoff_agent1_from_agent4.md`.

## Provenance

- A15 (audit): `CLAUDE_PHI_DATA_ANALYSIS/analyses/15_broadrange_retention/`
- A16 (v3 derivation + networks): `CLAUDE_PHI_DATA_ANALYSIS/analyses/16_multitop_hc/`
- Filter recipe: `classify_multi_top` in `analyses/16_multitop_hc/run.py`
- Network PNGs at edge ≥ 0.50 (v2-comparable), ≥ 0.33 (reveals K multi-bridges),
  and ≥ 0.80/0.90 for O — all in `analyses/16_multitop_hc/output/`.
