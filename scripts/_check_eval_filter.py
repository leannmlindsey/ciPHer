"""One-off — verify the new eval-side filter logic in per_head_strict_eval.py
produces protein-count totals matching leann's broadcast table:

    | dataset       | ALL  | pipeline_positive | any3strong | any4strong |
    | CHEN          |  30  |   13              |   23       |   23       |
    | GORODNICHIV   |  21  |   15              |   18       |   18       |
    | PBIP          | 681  |  312              |  539       |  541       |
    | PhageHostLearn| 483  |  256              |  377       |  379       |
    | UCSD          | 118  |   48              |   92       |   92       |
    | TOTAL         |1,333 |  644              | 1,049      | 1,053      |

Total counts (no per-dataset breakdown) are easy to verify against the
new validation glycan_binders TSV directly. Per-dataset breakdown
requires joining to phage_protein_mapping.csv per dataset.

Run: python scripts/_check_eval_filter.py
"""

import sys
from pathlib import Path

# Bring cipher.data on the path
HERE = Path(__file__).resolve().parent
SRC = HERE.parent / 'src'
sys.path.insert(0, str(SRC))

from cipher.data.proteins import load_glycan_binders, load_positive_list

GB_PATH = '/Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/validation_inputs/glycan_binders_custom.tsv'
PIPE_POS = '/Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/klebsiella/validation_data/combined/pipeline_pos/phagehostlearn_pipeline_positive.list'

print(f'[load] {GB_PATH}')
gb = load_glycan_binders(GB_PATH)
all_pids = set(gb.keys())
print(f'  {len(all_pids):,} total proteins (expected 1,333)')

# any3strong: SpikeHunter, DePP_85, PhageRBPdetect
any3 = {p for p in all_pids
        if any(int(gb[p].get(t, 0)) == 1 for t in ['SpikeHunter', 'DePP_85', 'PhageRBPdetect'])}
print(f'  any3strong: {len(any3):,} (expected 1,049)')

# any4strong: + DepoScope
any4 = {p for p in all_pids
        if any(int(gb[p].get(t, 0)) == 1 for t in ['SpikeHunter', 'DePP_85', 'PhageRBPdetect', 'DepoScope'])}
print(f'  any4strong: {len(any4):,} (expected 1,053)')

# Total source distribution
src_dist = {}
for p in all_pids:
    s = int(gb[p].get('total_sources', 0))
    src_dist[s] = src_dist.get(s, 0) + 1
print(f'  source distribution: {sorted(src_dist.items())}')
