"""Per-tool quality analysis.

For each tool, compare proteins flagged EXCLUSIVELY by that tool vs proteins
flagged by that tool AND at least one other. Measures:

1. How many exclusive vs shared proteins per tool
2. K-type specificity: exclusive proteins → how many K-types per protein?
   (noisy proteins tend to map to many K-types; clean ones map to 1-2)
3. Training representation: are exclusive proteins in the training data
   (i.e., do they appear in host_phage_protein_map.tsv)?
4. For proteins in training: mean #distinct K-types per protein
   (exclusive vs shared)

Outputs:
    data_exploration/output/tool_quality_summary.csv
    data_exploration/output/tool_quality_summary.txt
"""

import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import REPO_ROOT, OUTPUT_DIR

TOOLS = [
    'DePP_85', 'PhageRBPdetect', 'DepoScope', 'DepoRanker',
    'SpikeHunter', 'dbCAN', 'IPR', 'phold_glycan_tailspike',
]

SHORT = {
    'DePP_85': 'DePP', 'PhageRBPdetect': 'RBPdetect',
    'DepoScope': 'DepoScope', 'DepoRanker': 'DepoRanker',
    'SpikeHunter': 'SpikeHunter', 'dbCAN': 'dbCAN',
    'IPR': 'IPR', 'phold_glycan_tailspike': 'phold',
}

GLYCAN_PATH = os.path.join(
    REPO_ROOT, 'data', 'training_data', 'metadata', 'glycan_binders_custom.tsv')
ASSOC_PATH = os.path.join(
    REPO_ROOT, 'data', 'training_data', 'metadata', 'host_phage_protein_map.tsv')


def load_tool_flags_per_protein():
    """Returns {protein_id: set of tools that flagged it}."""
    prot_tools = {}
    with open(GLYCAN_PATH) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            pid = row['protein_id']
            flagged = set(t for t in TOOLS if int(row.get(t, 0)) == 1)
            if flagged:
                prot_tools[pid] = flagged
    return prot_tools


def load_training_associations():
    """Returns {protein_id: Counter({K_type: count})}."""
    from cipher.data.interactions import load_training_map
    rows = load_training_map(ASSOC_PATH)
    pid_k = defaultdict(Counter)
    for r in rows:
        pid_k[r['protein_id']][r['host_K']] += 1
    return dict(pid_k)


def main():
    print('Loading tool flags...')
    prot_tools = load_tool_flags_per_protein()
    print(f'  {len(prot_tools)} proteins with tool flags')

    print('Loading training associations...')
    pid_k = load_training_associations()
    training_pids = set(pid_k.keys())
    print(f'  {len(training_pids)} proteins in training association table')

    results = []

    for tool in TOOLS:
        # Proteins flagged by this tool
        tool_pids = {p for p, ts in prot_tools.items() if tool in ts}

        # Exclusive: flagged by ONLY this tool
        exclusive = {p for p in tool_pids if len(prot_tools[p]) == 1}

        # Shared: flagged by this tool + at least one other
        shared = tool_pids - exclusive

        # How many are in training data?
        excl_in_train = exclusive & training_pids
        shared_in_train = shared & training_pids

        # K-type specificity for those in training
        def k_stats(pid_set):
            n_k_list = []
            for pid in pid_set:
                if pid in pid_k:
                    n_k_list.append(len(pid_k[pid]))
            if not n_k_list:
                return 0, 0, 0, 0
            arr = np.array(n_k_list)
            return len(n_k_list), float(arr.mean()), float(np.median(arr)), int(arr.max())

        excl_n, excl_mean_k, excl_med_k, excl_max_k = k_stats(excl_in_train)
        shared_n, shared_mean_k, shared_med_k, shared_max_k = k_stats(shared_in_train)

        # Fraction with only 1 K-type (= clean, specific signal)
        excl_single_k = sum(1 for p in excl_in_train if p in pid_k and len(pid_k[p]) == 1)
        shared_single_k = sum(1 for p in shared_in_train if p in pid_k and len(pid_k[p]) == 1)

        row = {
            'tool': SHORT[tool],
            'total': len(tool_pids),
            'exclusive': len(exclusive),
            'shared': len(shared),
            'excl_pct': f'{100 * len(exclusive) / len(tool_pids):.1f}%',
            'excl_in_training': len(excl_in_train),
            'shared_in_training': len(shared_in_train),
            'excl_mean_n_ktypes': f'{excl_mean_k:.2f}' if excl_n else 'N/A',
            'shared_mean_n_ktypes': f'{shared_mean_k:.2f}' if shared_n else 'N/A',
            'excl_median_n_ktypes': f'{excl_med_k:.0f}' if excl_n else 'N/A',
            'shared_median_n_ktypes': f'{shared_med_k:.0f}' if shared_n else 'N/A',
            'excl_single_k_pct': f'{100 * excl_single_k / max(excl_n, 1):.1f}%',
            'shared_single_k_pct': f'{100 * shared_single_k / max(shared_n, 1):.1f}%',
        }
        results.append(row)

    # CSV output
    csv_path = os.path.join(OUTPUT_DIR, 'tool_quality_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f'\nSaved: {csv_path}')

    # Print summary
    lines = []
    lines.append('=' * 90)
    lines.append('TOOL QUALITY ANALYSIS: Exclusive vs Shared Proteins')
    lines.append('=' * 90)
    lines.append('')
    lines.append(f'{"Tool":<13} {"Total":>7} {"Excl":>7} {"Shared":>7} {"Excl%":>6} '
                 f'{"ExTrain":>8} {"ShTrain":>8} '
                 f'{"Ex K/prot":>9} {"Sh K/prot":>9} '
                 f'{"Ex 1K%":>7} {"Sh 1K%":>7}')
    lines.append('-' * 90)
    for r in results:
        lines.append(
            f'{r["tool"]:<13} {r["total"]:>7} {r["exclusive"]:>7} {r["shared"]:>7} '
            f'{r["excl_pct"]:>6} '
            f'{r["excl_in_training"]:>8} {r["shared_in_training"]:>8} '
            f'{r["excl_mean_n_ktypes"]:>9} {r["shared_mean_n_ktypes"]:>9} '
            f'{r["excl_single_k_pct"]:>7} {r["shared_single_k_pct"]:>7}')

    lines.append('')
    lines.append('Columns:')
    lines.append('  Excl        = proteins flagged by ONLY this tool (no other tool)')
    lines.append('  Shared      = proteins flagged by this tool + at least 1 other')
    lines.append('  ExTrain     = exclusive proteins that appear in training associations')
    lines.append('  ShTrain     = shared proteins in training associations')
    lines.append('  Ex K/prot   = mean #distinct K-types per exclusive protein (in training)')
    lines.append('  Sh K/prot   = mean #distinct K-types per shared protein (in training)')
    lines.append('  Ex 1K%      = % of exclusive training proteins with exactly 1 K-type')
    lines.append('  Sh 1K%      = % of shared training proteins with exactly 1 K-type')
    lines.append('')
    lines.append('Interpretation:')
    lines.append('  If exclusive proteins have MORE K-types per protein (lower specificity)')
    lines.append('  and FEWER in training, they may be low-quality predictions adding noise.')

    summary = '\n'.join(lines)
    print('\n' + summary)

    txt_path = os.path.join(OUTPUT_DIR, 'tool_quality_summary.txt')
    with open(txt_path, 'w') as f:
        f.write(summary + '\n')
    print(f'\nSaved: {txt_path}')


if __name__ == '__main__':
    main()
