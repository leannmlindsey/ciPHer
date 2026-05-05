"""Quick check — K-only vs O-only vs OR HR@1 on PHL and PBIP for the
two headline models. Tests: is PBIP's 0.98 HR@1 driven by K, O, or both?
Implication: if PBIP O@1 is much higher than its K@1, the O-head is
'rescuing' PBIP from the same K-label-disagreement issue PHL suffers.
"""
import csv
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TSVS = {
    'sweep_kmer_aa20_k4 (best single)':            'sweep_kmer_aa20_k4',
    'sweep_posList_esm2_3b_mean_cl70 (hybrid K)':  'sweep_posList_esm2_3b_mean_cl70',
    'sweep_prott5_mean_cl70':                       'sweep_prott5_mean_cl70',
}
N_FIXED = {'PhageHostLearn': 100, 'PBIP': 103}

print(f'{"model":<48s} {"dataset":<14s} {"K@1":>6s} {"O@1":>6s} {"OR@1":>6s}')
print('-' * 90)
for name, run in TSVS.items():
    path = REPO / 'results' / 'analysis' / 'per_phage' / f'per_phage_{run}.tsv'
    if not path.exists():
        print(f'  MISSING: {path}')
        continue
    rows_by_ds = {}
    for r in csv.DictReader(open(path), delimiter='\t'):
        rows_by_ds.setdefault(r['dataset'], []).append(r)
    for ds in ['PhageHostLearn', 'PBIP']:
        rs = rows_by_ds.get(ds, [])
        denom = N_FIXED[ds]
        k = sum(1 for r in rs if r.get('k_hit@1') == '1') / denom
        o = sum(1 for r in rs if r.get('o_hit@1') == '1') / denom
        orh = sum(1 for r in rs if r.get('or_hit@1') == '1') / denom
        print(f'{name:<48s} {ds:<14s} {k:6.3f} {o:6.3f} {orh:6.3f}')
    print()
