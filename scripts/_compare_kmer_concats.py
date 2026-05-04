"""Compare every kmer-related concat run to the standalone baselines and
to each other. Goal: explain why concat_prott5_mean+kmer_aa20_k4 (0.34)
fails when concat_prott5_mean+kmer_li10_k5 (0.48) and concat_esm2_3b_mean
+kmer_aa20_k4 (0.53) work.
"""
import csv
from collections import defaultdict

CSV = 'results/experiment_log.csv'
FIXED = {'CHEN':3, 'GORODNICHIV':3, 'UCSD':11, 'PBIP':103, 'PhageHostLearn':100}

def f(v):
    try: return float(v) if v else None
    except (TypeError, ValueError): return None

def hits(r, ds, mode='OR', k=1):
    n = f(r.get(f'{ds}_n_strict_phage'))
    hr = f(r.get(f'{ds}_{mode}_phage2host_anyhit_HR{k}'))
    return round(hr*n) if (n is not None and hr is not None) else None

def hrk(r, ds, mode='OR', k=1):
    h = hits(r, ds, mode, k); return h/FIXED[ds] if h is not None else None

rows = list(csv.DictReader(open(CSV)))

# 1. All KMER STANDALONES — establish baseline performance per kmer variant
kmer_standalones = sorted(
    [r for r in rows if r['run_name'].startswith('sweep_kmer_') and not r.get('embedding_type_2','').strip()],
    key=lambda r: -(hrk(r,'PhageHostLearn') or 0)
)
def fmt(x): return f'{x:.3f}' if x is not None else '  -- '

print('═══ Kmer STANDALONE runs (no concat) — sorted by PHL OR HR@1 ═══')
print(f'  {"PHL":>5s}  {"PHL_K":>5s}  {"PHL_O":>5s}  {"emb":<22s}  {"n_md5":>6s}  {"K-cls":>5s}  run_name')
for r in kmer_standalones:
    print(f'  {fmt(hrk(r,"PhageHostLearn")):>5s}  {fmt(hrk(r,"PhageHostLearn","K")):>5s}  '
          f'{fmt(hrk(r,"PhageHostLearn","O")):>5s}  {r.get("embedding_type",""):<22s}  '
          f'{r.get("n_md5s",""):>6s}  {r.get("n_k_classes",""):>5s}  {r["run_name"]}')

# 2. ALL CONCAT runs — sorted by PHL OR HR@1
concat_runs = sorted(
    [r for r in rows if r.get('embedding_type_2','').strip() or '+' in r['run_name']],
    key=lambda r: -(hrk(r,'PhageHostLearn') or 0)
)
print()
print('═══ All CONCAT runs (pLM + kmer) — sorted by PHL OR HR@1 ═══')
print(f'  {"PHL":>5s}  {"PHL_K":>5s}  {"PHL_O":>5s}  {"pLM":<18s}  {"kmer":<22s}  {"max_K":>5s} {"max_O":>5s}  {"label_strat":<22s}  run')
for r in concat_runs:
    e1 = r.get('embedding_type','')
    e2 = r.get('embedding_type_2','') or 'inferred'
    print(f'  {fmt(hrk(r,"PhageHostLearn"))}  {fmt(hrk(r,"PhageHostLearn","K"))}  '
          f'{fmt(hrk(r,"PhageHostLearn","O"))}  {e1:<18s}  {e2:<22s}  '
          f'{r.get("max_samples_per_k",""):>5s} {r.get("max_samples_per_o",""):>5s}  '
          f'{r.get("label_strategy",""):<22s}  {r["run_name"]}')

# 3. Dimensional / training detail comparison for the 3 head-to-head concats
print()
print('═══ Head-to-head: 4 concats with kmer_aa20_k4 vs kmer_li10_k5 ═══')
key_runs = [
    'sweep_kmer_aa20_k4',                              # aa20 kmer alone
    'sweep_kmer_li10_k5',                              # li10 kmer alone (if exists)
    'concat_prott5_mean+kmer_aa20_k4',                 # prott5 + aa20 (FAILS at 0.34)
    'concat_prott5_mean+kmer_li10_k5',                 # prott5 + li10 (works at 0.48)
    'concat_esm2_3b_mean+kmer_aa20_k4',                # 3b    + aa20 (works at 0.53)
    'concat_esm2_3b_mean+kmer_li10_k5',                # 3b    + li10 (recent — check)
    'concat_esm2_3b_mean+kmer_murphy8_k5',             # 3b    + murphy8
    'concat_esm2_650m_seg4+kmer_li10_k5',              # 650m seg4 + li10
]
print(f'  {"PHL_OR":>7s}  {"PHL_K":>6s}  {"PHL_O":>6s}  {"PBIP_OR":>7s}  {"emb1":<18s}  {"emb2":<22s}  '
      f'{"max_K":>5s} {"max_O":>5s}  {"lr":>8s}  {"min_class":>9s}  run')
for run in key_runs:
    r = next((x for x in rows if x['run_name']==run), None)
    if r is None:
        print(f'  -- not in CSV --                                                                          {run}')
        continue
    print(f'  {fmt(hrk(r,"PhageHostLearn")):>7s}  {fmt(hrk(r,"PhageHostLearn","K")):>6s}  '
          f'{fmt(hrk(r,"PhageHostLearn","O")):>6s}  {fmt(hrk(r,"PBIP")):>7s}  '
          f'{(r.get("embedding_type","")+"        ")[:18]}  '
          f'{(r.get("embedding_type_2","")+"                  ")[:22]}  '
          f'{r.get("max_samples_per_k",""):>5s} {r.get("max_samples_per_o",""):>5s}  '
          f'{r.get("lr",""):>8s}  {r.get("min_class_samples",""):>9s}  {run}')
