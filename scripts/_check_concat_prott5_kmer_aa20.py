"""One-off diagnostic — check whether the concat_prott5_mean+kmer_aa20_k4
training pipeline actually completed and wrote the eval artifacts.

Run on Delta from the cipher root:
    python scripts/_check_concat_prott5_kmer_aa20.py

Prints which of the 4 expected files exist and what they contain.
"""

import json
import os
from pathlib import Path

run = 'concat_prott5_mean+kmer_aa20_k4'
exp = Path(f'experiments/attention_mlp/{run}')
tsv = Path(f'results/analysis/per_phage/per_phage_{run}.tsv')

files = {
    'config.yaml':                 exp / 'config.yaml',
    'experiment.json':             exp / 'experiment.json',
    'model_k/best_model.pt':       exp / 'model_k' / 'best_model.pt',
    'model_o/best_model.pt':       exp / 'model_o' / 'best_model.pt',
    'results/evaluation.json':     exp / 'results' / 'evaluation.json',
    'results/per_head_strict_eval.json': exp / 'results' / 'per_head_strict_eval.json',
    'per_phage TSV (results/analysis/)': tsv,
}

print(f'=== {run} — file inventory ===')
for label, p in files.items():
    if p.exists():
        size = p.stat().st_size
        print(f'  OK     {size:>12,d} B   {label}')
    else:
        print(f'  MISSING                {label}')

# If per_head_strict_eval.json exists, summarize its PHL numbers
peh = exp / 'results' / 'per_head_strict_eval.json'
if peh.exists():
    print()
    print(f'=== {peh} — PHL summary ===')
    d = json.load(open(peh))
    phl = d.get('PhageHostLearn', {})
    for k in ('n_strict_phage', 'n_strict_pair', 'or_anyhit_HR1',
              'best_anyhit_HR1', 'best_strict_HR1'):
        print(f'  {k} = {phl.get(k)}')
