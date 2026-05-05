"""Print a concrete-examples table of PHL-miss phages whose highest-id
training neighbour has a mismatched K-label, with the model's actual
top-1 predicted K-class added as a column. For PI-meeting slides.

Sources:
  - agent 4's analysis 32 drilldown CSV (closest training neighbour per RBP)
  - per_phage_top1_sweep_prott5_mean_cl70.tsv (model's top-1 predicted K)

Note: top-1 dump is currently only available for sweep_prott5_mean_cl70;
the headline esm2_3b model's per_phage_top1 isn't dumped yet (agent 4's
2026-05-04 ask). prott5 and esm2_3b agree on ~88% of PHL phages per
2026-05-04_pLM_redundancy_phl.md, so prott5 top-1 is a good proxy.
"""
import csv
from collections import defaultdict
from pathlib import Path

CSV = Path('/Users/leannmlindsey/WORK/CLAUDE_PHI_DATA_ANALYSIS/'
           'analyses/32_per_serotype_model_comparison/output/'
           'phl_miss_neighbour_drilldown.csv')
TOP1 = Path('/Users/leannmlindsey/WORK/PHI_TSP/cipher/results/analysis/'
            'per_phage_top1/per_phage_top1_sweep_prott5_mean_cl70.tsv')

rows = list(csv.DictReader(open(CSV)))

# Load model top-1 predictions per phage (PhageHostLearn only)
phage_top1 = {}
with open(TOP1) as fh:
    for r in csv.DictReader(fh):
        if r.get('dataset') == 'PhageHostLearn':
            phage_top1[r['phage_id']] = r.get('cp_top1_set', '')

# Per (phage, RBP) keep the single highest-id training neighbour
best = {}
for r in rows:
    key = (r['phl_phage'], r['phl_rbp_id'])
    pid = float(r['pident'])
    if key not in best or pid > float(best[key]['pident']):
        best[key] = r

# Per phage keep the single highest-id RBP↔neighbour pair
phage_best = {}
for (ph, rbp), r in best.items():
    if ph not in phage_best or float(r['pident']) > float(phage_best[ph]['pident']):
        phage_best[ph] = r

# Sort phages by best-pair pident (descending) to surface the most striking
ordered = sorted(phage_best.values(), key=lambda r: -float(r['pident']))

print(f'PHL phages we MISS, with the closest training-neighbour K-label')
print(f'(sorted by sequence identity — top of list = most striking same-sequence-different-label cases)\n')

print(f'{"PHL phage":<22s} {"phage host K":<25s} {"%id":>6s}  {"training K":<18s} {"prott5 predicted K":<22s} {"agreement":<32s}')
print('-' * 130)
n_pred_matches_train = 0
n_examples_with_pred = 0
for r in ordered[:30]:
    phage = r['phl_phage']
    phl_K_raw = r['phl_full_K_set'] or r['phl_dominant_K']
    phl_K_set = set(s for s in phl_K_raw.split(';') if s)
    pident = float(r['pident'])
    train_K_raw = r['train_K_labels'] or ''
    train_K_set = set(s for s in train_K_raw.split(';') if s)
    pred_raw = phage_top1.get(phage, '')
    pred_set = set(s for s in pred_raw.split(';') if s)

    # Normalize KL<n> ↔ K<n> (Kaptive's "KL" prefix = the same K serotype)
    def normK(k):
        return ('K' + k[2:]) if k.startswith('KL') and k[2:].isdigit() else k
    phl_K_norm = {normK(k) for k in phl_K_set}
    train_K_norm = {normK(k) for k in train_K_set}
    pred_norm = {normK(k) for k in pred_set}

    pred_matches_phage = bool(pred_norm & phl_K_norm)
    pred_matches_train = bool(pred_norm & train_K_norm)

    if pred_set:
        n_examples_with_pred += 1
        if pred_matches_train and not pred_matches_phage:
            n_pred_matches_train += 1
            agreement_label = 'predicted = TRAINING (the trap)'
        elif pred_matches_phage:
            agreement_label = 'predicted = PHAGE host (rescue)'
        else:
            agreement_label = 'predicted = neither'
    else:
        agreement_label = '(no prediction)'

    print(f'{phage:<22s} {phl_K_raw[:25]:<25s} {pident:>5.1f}%  '
          f'{train_K_raw[:18]:<18s} {pred_raw[:22]:<22s} {agreement_label:<32s}')

print()
print(f'Of {n_examples_with_pred} examples (top 30) with a prott5 prediction:')
print(f'  {n_pred_matches_train} ({100*n_pred_matches_train/n_examples_with_pred:.0f}%) '
      f'have model predicted = training-neighbour K (the model fell into the trap)')

print()
n_disagree = sum(1 for r in ordered if r['train_matches_any_phl_K'] != 'yes')
print(f'Of {len(ordered)} PHL-miss phages with high-id (≥80%) training neighbours, '
      f'{n_disagree} ({100*n_disagree/len(ordered):.0f}%) have their CLOSEST '
      f'training neighbour carrying a different K-label.')
