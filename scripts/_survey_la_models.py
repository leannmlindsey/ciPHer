"""One-off — survey light_attention models in experiment_log.csv,
sorted by PhageHostLearn OR HR@1 (the headline metric).

Run: python scripts/_survey_la_models.py
"""

import csv

CSVS = [
    ('/Users/leannmlindsey/WORK/PHI_TSP/cipher-light-attention/results/experiment_log.csv', 'cipher-light-attention'),
    ('/Users/leannmlindsey/WORK/PHI_TSP/cipher-light-attention-binary/results/experiment_log.csv', 'cipher-light-attention-binary'),
    ('/Users/leannmlindsey/WORK/PHI_TSP/cipher-binary-onevsrest/results/experiment_log.csv', 'cipher-binary-onevsrest'),
]

la = []
for csv_path, worktree in CSVS:
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        continue
    print(f'  {worktree}: {len(rows)} rows total in experiment_log')
    for r in rows:
        r['_worktree'] = worktree
    la.extend(rows)
print(f'\nTotal rows across all 3 LA-related worktrees: {len(la)}\n')

def to_f(v):
    try: return float(v) if v else None
    except (TypeError, ValueError): return None

# Sort by PhageHostLearn OR_phage2host_anyhit_HR1 desc
def sort_key(r):
    return -(to_f(r.get('PhageHostLearn_OR_phage2host_anyhit_HR1')) or 0)

la.sort(key=sort_key)

print(f'{"PHL_OR":>6s} {"PHL_K":>6s} {"PHL_O":>6s} {"PBIP_OR":>7s} {"worktree":<28s} {"emb":<22s} run_name')
for r in la[:25]:
    phl_or = to_f(r.get('PhageHostLearn_OR_phage2host_anyhit_HR1'))
    phl_k = to_f(r.get('PhageHostLearn_K_phage2host_anyhit_HR1'))
    phl_o = to_f(r.get('PhageHostLearn_O_phage2host_anyhit_HR1'))
    pbip_or = to_f(r.get('PBIP_OR_phage2host_anyhit_HR1'))
    fmt = lambda x: f'{x:.3f}' if x is not None else '   --'
    print(f'{fmt(phl_or):>6s} {fmt(phl_k):>6s} {fmt(phl_o):>6s} {fmt(pbip_or):>7s} '
          f'{(r.get("_worktree","") or "")[:28]:<28s} {(r.get("embedding_type","") or "")[:22]:<22s} '
          f'{r.get("run_name","")}')
