# Final highconf training lists — for independent per-head training

These are the four deliverable protein-ID lists for the new
**per-head independent training** design:

- K head loss is computed only on proteins in the K list
- O head loss is computed only on proteins in the O list
- Each protein still feeds through the shared encoder once; the change
  is loss-masking per head

## Files

| File | Protein count | Serotypes covered | Notes |
|---|---|---|---|
| `HC_K_cl95.list` | 23,299 | 161 / 161 K-types | Strict: each protein's 95% cluster is ≥ 90% pure for K |
| `HC_K_UAT.list` | 25,924 | 161 / 161 K-types | Union across thresholds 60/70/80/85/90/95 — more data, looser purity |
| `HC_O_cl95_full_coverage.list` | 14,677 | 22 / 22 O-types | Strict cl95 + 13-protein rescue for `O11α,2β` |
| `HC_O_UAT.list` | 15,568 | 22 / 22 O-types | Union across thresholds — more data, looser purity |

## Two variants to train on

- **Strict pair**: `HC_K_cl95.list` + `HC_O_cl95_full_coverage.list`
- **UAT pair**:    `HC_K_UAT.list`  + `HC_O_UAT.list`

Train both and compare PHL rh@1 / hr@5 to decide which recipe wins.

## How the filter works

1. Input: `pipeline_positive.list` (59,182 proteins after intersecting
   with the association map)
2. For each CD-HIT clustering threshold T (60 / 70 / 80 / 85 / 90 / 95):
   - Compute SpikeHunter-style cluster signals (dedup key:
     `(c95, phage_id, serotype)`) separately for host_K and host_O
   - A cluster is "highly confident" for axis X if the top X-serotype
     accounts for ≥ 90% of deduped supports (or 100% for clusters of
     size ≤ 4)
3. `HC_X_clT` = pipeline_positive proteins whose T% cluster is HC-for-X
4. `HC_X_UAT` = union of `HC_X_clT` across all T

`HC_O_cl95_full_coverage.list` also includes the 13 pipeline_positive
proteins labeled `O11α,2β`, which drop out of every single-threshold
HC filter. Without them, the strict-cl95 variant would lose that
O-type entirely (only 13 pipeline_positive proteins carry it).

## Under-served serotypes (< 5 proteins retained)

Flagged for awareness — the model will have limited learning signal
for these, but they are represented.

**Strict cl95:**
- K: K44 (1), KL175 (1), KL144 (4)
- O: `O11α,2β` (13, rescued)

**UAT:**
- K: K44 (3), KL144 (4), KL175 (1)
- O: `O11α,2β` (13)

## Provenance

- Analysis: `CLAUDE_PHI_DATA_ANALYSIS/analyses/13_optimal_highconf/`
- Coverage CSVs per variant: `HC_{K,O}_{cl60,cl70,cl80,cl85,cl90,cl95,UAT}_coverage.csv`
- Full threshold sweep: `threshold_sweep_summary.csv` (this directory's parent)
- Filter recipe implemented in `src/cipher_analysis/clusters.py::classify_pair`
  (`highly_confident` branch only)
