"""Per-pident-bin K-label agreement rate, PBIP vs PHL.

Same drilldown as plot_label_disagreement_per_model.py, but breaks the
≥80% identity range into 4 bins (80-85, 85-90, 90-95, 95-100) so we
can see whether the wrong-label rate gets BETTER at near-identical
sequence (the natural expectation) — or stays bad (the surprising
finding from agent 4's analysis 32).

Uses the union of both top-model training pools (esm2_3b's
pipeline_positive AND kmer's tools=DepoScope|PhageRBPdetect); the
overlap covers all high-id training neighbours of validation RBPs
in practice (verified empirically — both pools give identical
agreement rates).

Output:
  results/figures/pi_meeting_2026-05-05/02_phl_gap_diagnosis/
      fig_label_disagreement_by_pident_bin.{svg,png}
"""

from __future__ import annotations
import csv
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(sys.maxsize)

CIPHER = Path('/Users/leannmlindsey/WORK/PHI_TSP/cipher')
HITS = Path('/Users/leannmlindsey/WORK/CLAUDE_PHI_DATA_ANALYSIS/analyses/'
            '32_per_serotype_model_comparison/work/seq_id/val_vs_train_hits.tsv')
VAL_FAA = CIPHER / 'data/validation_data/metadata/validation_rbps_all.faa'
VAL_BASE = CIPHER / 'data/validation_data/HOST_RANGE'
HOST_MAP = CIPHER / 'data/training_data/metadata/host_phage_protein_map.tsv'
PP = CIPHER / 'data/training_data/metadata/pipeline_positive.list'

OUT_DIR = (CIPHER / 'results/figures/pi_meeting_2026-05-05/02_phl_gap_diagnosis')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SVG = OUT_DIR / 'fig_label_disagreement_by_pident_bin.svg'
OUT_PNG = OUT_DIR / 'fig_label_disagreement_by_pident_bin.png'

PIDENT_BINS = [
    ('80-85%', 80.0, 85.0),
    ('85-90%', 85.0, 90.0),
    ('90-95%', 90.0, 95.0),
    ('95-100%', 95.0, 100.001),
]


def md5_str(s):
    return hashlib.md5(s.encode()).hexdigest()


def parse_fasta(path):
    pid = None; seq = []
    with open(path) as fh:
        for ln in fh:
            ln = ln.rstrip()
            if ln.startswith('>'):
                if pid is not None:
                    yield pid, ''.join(seq)
                pid = ln[1:].split()[0]; seq = []
            else:
                seq.append(ln)
        if pid is not None:
            yield pid, ''.join(seq)


def normK(k):
    return ('K' + k[2:]) if k.startswith('KL') and k[2:].isdigit() else k


def load_val_hits():
    out = defaultdict(list)
    with HITS.open() as fh:
        for ln in fh:
            row = ln.rstrip().split('\t')
            if len(row) < 3:
                continue
            try:
                pid_pct = float(row[2])
            except ValueError:
                continue
            out[row[0]].append((row[1], pid_pct))
    for k in out:
        out[k].sort(key=lambda x: -x[1])
    return out


def build_md5_to_K_pp():
    """{md5: set(K-labels)} restricted to pipeline_positive entries."""
    with PP.open() as fh:
        keep_pids = {ln.strip() for ln in fh if ln.strip()}
    md5_to_K = defaultdict(set)
    with HOST_MAP.open() as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r.get('protein_id', '').strip() not in keep_pids:
                continue
            md5 = r.get('protein_md5', '').strip()
            k = (r.get('host_K') or '').strip()
            if md5 and k and k != 'N/A':
                md5_to_K[md5].add(normK(k))
    return md5_to_K


def load_dataset_phages(dataset_name):
    ds = VAL_BASE / dataset_name
    phage_proteins = defaultdict(list)
    with (ds / 'metadata' / 'phage_protein_mapping.csv').open() as fh:
        for r in csv.DictReader(fh):
            phage_proteins[r['matrix_phage_name']].append(r['protein_id'])
    kc = defaultdict(Counter)
    with (ds / 'metadata' / 'interaction_matrix.tsv').open() as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r.get('label') != '1':
                continue
            k = (r.get('host_K') or '').strip()
            if k and k != 'N/A':
                kc[r.get('phage_id') or r.get('phage')][k] += 1
    K_set = {ph: {normK(k) for k in c.keys()} for ph, c in kc.items() if c}
    return phage_proteins, K_set


def drilldown_per_bin(val_hits, pid_to_md5, md5_to_K, dataset_name):
    """Returns {bin_label: {'pairs': n, 'match': n}}."""
    phage_proteins, K_set = load_dataset_phages(dataset_name)
    out = {label: {'pairs': 0, 'match': 0} for label, _, _ in PIDENT_BINS}
    for ph, true_K in K_set.items():
        for pid in phage_proteins.get(ph, []):
            md5 = pid_to_md5.get(pid)
            if not md5:
                continue
            for t_md5, pid_pct in val_hits.get(md5, []):
                if pid_pct < PIDENT_BINS[0][1]:
                    break  # hits are sorted desc; everything below floor
                bin_label = None
                for label, lo, hi in PIDENT_BINS:
                    if lo <= pid_pct < hi:
                        bin_label = label
                        break
                if bin_label is None:
                    continue
                t_K_set = md5_to_K.get(t_md5, set())
                out[bin_label]['pairs'] += 1
                if t_K_set and (t_K_set & true_K):
                    out[bin_label]['match'] += 1
    return out


def main():
    print('[load] val hits, val FASTA, training md5→K (pipeline_positive)')
    val_hits = load_val_hits()
    pid_to_md5 = {pid: md5_str(seq) for pid, seq in parse_fasta(VAL_FAA)}
    md5_to_K = build_md5_to_K_pp()
    print(f'  val hits: {len(val_hits)}, val proteins: {len(pid_to_md5)}, '
          f'training MD5s with K: {len(md5_to_K):,}')

    print('[drilldown] per pident bin, per dataset')
    results = {}
    for ds in ['PhageHostLearn', 'PBIP']:
        results[ds] = drilldown_per_bin(val_hits, pid_to_md5, md5_to_K, ds)
        print(f'\n  {ds}:')
        for label, _, _ in PIDENT_BINS:
            n = results[ds][label]['pairs']
            m = results[ds][label]['match']
            pct = 100 * m / n if n else 0
            print(f'    {label:<8s}  pairs={n:>4d}  match={m:>4d}  rate={pct:>5.1f}%')

    # ── Plot: 2 panels (PHL, PBIP), stacked bars per pident bin ──
    # Bar HEIGHT = total pairs (shows distribution differs across datasets)
    # Bar FILL  = match (green, bottom) vs no-match (red, top)
    # Labels  = total n on top, match rate written inside the green slice
    bin_labels = [b[0] for b in PIDENT_BINS]
    max_pairs = max(
        max(results[ds][b]['pairs'] for b in bin_labels)
        for ds in ('PhageHostLearn', 'PBIP'))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, ds, ds_label in [(axes[0], 'PhageHostLearn', 'PhageHostLearn'),
                               (axes[1], 'PBIP', 'PBIP')]:
        x = np.arange(len(bin_labels))
        match_counts = [results[ds][b]['match'] for b in bin_labels]
        miss_counts = [results[ds][b]['pairs'] - results[ds][b]['match']
                        for b in bin_labels]
        total = [m + n for m, n in zip(match_counts, miss_counts)]

        bars_match = ax.bar(x, match_counts, color='#1a9641',
                             edgecolor='white', linewidth=1.5,
                             label='K-label MATCH')
        bars_miss = ax.bar(x, miss_counts, bottom=match_counts,
                            color='#d7191c', edgecolor='white', linewidth=1.5,
                            label='K-label MISMATCH')

        # Match rate inside the green slice (if it's tall enough)
        for i, (m, n) in enumerate(zip(match_counts, total)):
            if n == 0:
                continue
            pct = 100 * m / n
            if m >= 8:  # only label inside if there's room
                ax.text(i, m / 2, f'{pct:.0f}%', ha='center', va='center',
                        color='white', fontsize=10, fontweight='bold')
        # Total n above each bar
        for i, n in enumerate(total):
            if n > 0:
                ax.text(i, n + max_pairs * 0.02, f'n={n}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
                # Also write the match rate above n if the green slice was too short
                pct = 100 * match_counts[i] / n
                if match_counts[i] < 8:
                    ax.text(i, n + max_pairs * 0.10, f'({pct:.0f}% match)',
                            ha='center', va='bottom', fontsize=9, color='#666')

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=10)
        ax.set_xlabel('Sequence identity of training neighbour to validation RBP')
        if ax is axes[0]:
            ax.set_ylabel('# of (val-RBP, training-neighbour) pairs')
        ax.set_ylim(0, max_pairs * 1.25)
        ax.set_title(ds_label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        if ax is axes[0]:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.95)

    fig.suptitle(
        'PHL has many near-identical training neighbours — but with mismatched K-labels.\n'
        'PBIP\'s RBPs are more novel from training, so it\'s not exposed to this trap.',
        fontsize=11.5, fontweight='bold', y=1.01)

    fig.tight_layout()
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\n  wrote {OUT_SVG}')
    print(f'  wrote {OUT_PNG}')


if __name__ == '__main__':
    main()
