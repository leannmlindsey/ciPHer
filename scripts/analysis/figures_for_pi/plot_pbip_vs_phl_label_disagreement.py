"""Compare the same-sequence-different-K-label rate between PHL and PBIP.

Hypothesis (from PI meeting prep): PBIP doesn't suffer the same
"high-id training neighbour with wrong K-label" problem that PHL does.
Either PBIP's RBPs happen to carry training-consistent K-labels, or
PBIP samples a different region of RBP sequence space.

Mirrors agent 4's `build_phl_miss_neighbour_drilldown.py` but:
  - Iterates ALL phages in BOTH PHL and PBIP (not just misses), since
    PBIP barely misses anything (~2 phages out of 103) and we need a
    bigger sample.
  - Computes agreement rate side-by-side and produces a comparison fig.

Inputs:
  - val_vs_train_hits.tsv (mmseqs results, agent 4)
  - validation_rbps_all.faa (all val RBP sequences → MD5)
  - HOST_RANGE/<DATASET>/metadata/{phage_protein_mapping.csv,
                                    interaction_matrix.tsv}
  - data/training_data/metadata/host_phage_protein_map.tsv (training MD5 → K)

Output: results/figures/pi_meeting_2026-05-05/02_phl_gap_diagnosis/
        fig_pbip_vs_phl_label_disagreement.{svg,png}
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
PIPELINE_POSITIVE = CIPHER / 'data/training_data/metadata/pipeline_positive.list'

OUT_DIR = (CIPHER / 'results/figures/pi_meeting_2026-05-05/02_phl_gap_diagnosis')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SVG = OUT_DIR / 'fig_pbip_vs_phl_label_disagreement.svg'
OUT_PNG = OUT_DIR / 'fig_pbip_vs_phl_label_disagreement.png'

PIDENT_FLOOR = 80.0  # only consider training neighbours at >=80% sequence id

# Restrict the training-K lookup to MD5s that are in pipeline_positive — i.e.,
# the actual training pool used by the headline ESM-2 3B K-source model. If
# False, use all training entries with K-labels (~59k, the broader association
# table).
RESTRICT_TO_PIPELINE_POSITIVE = True


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


def drilldown_for_dataset(dataset_dir, dataset_name, val_to_hits, pid_to_md5, md5_to_K):
    """Return list of dicts (one per (RBP, training-neighbour) pair) for ALL
    phages in the dataset (not just misses)."""
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
    phage_K_dom = {ph: c.most_common(1)[0][0] for ph, c in kc.items() if c}
    phage_K_set = {ph: {normK(k) for k in c.keys()} for ph, c in kc.items() if c}

    rows = []
    for ph in phage_K_set:
        true_K_set = phage_K_set[ph]
        true_K_dom = normK(phage_K_dom.get(ph, ''))
        for pid in phage_proteins.get(ph, []):
            md5 = pid_to_md5.get(pid)
            if not md5:
                continue
            hits = val_to_hits.get(md5, [])
            if not hits:
                continue
            top = [(t, p) for t, p in hits if p >= PIDENT_FLOOR][:5]
            for t_md5, pid_pct in top:
                t_K_set = md5_to_K.get(t_md5, set())
                rows.append({
                    'dataset': dataset_name,
                    'phage': ph, 'rbp_id': pid, 'pident': pid_pct,
                    'train_K_labels': ','.join(sorted(t_K_set)),
                    'matches_dom': true_K_dom in t_K_set,
                    'matches_any': bool(t_K_set & true_K_set),
                })
    return rows


def main():
    print('[1/4] loading mmseqs hits')
    val_to_hits = defaultdict(list)
    with HITS.open() as fh:
        for ln in fh:
            row = ln.rstrip().split('\t')
            if len(row) < 3:
                continue
            try:
                pid_pct = float(row[2])
            except ValueError:
                continue
            val_to_hits[row[0]].append((row[1], pid_pct))
    for v in val_to_hits:
        val_to_hits[v].sort(key=lambda x: -x[1])
    print(f'  {len(val_to_hits)} val MD5s with hits')

    print('[2/4] building val pid -> md5 from FASTA')
    pid_to_md5 = {pid: md5_str(seq) for pid, seq in parse_fasta(VAL_FAA)}
    print(f'  {len(pid_to_md5)} val proteins')

    print('[3/4] training MD5 -> K labels')
    keep_pids = None
    if RESTRICT_TO_PIPELINE_POSITIVE:
        with PIPELINE_POSITIVE.open() as fh:
            keep_pids = {ln.strip() for ln in fh if ln.strip()}
        print(f'  filtering to pipeline_positive: {len(keep_pids):,} training protein_ids')
    md5_to_K = defaultdict(set)
    n_kept_rows = 0
    with HOST_MAP.open() as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if keep_pids is not None and r.get('protein_id', '').strip() not in keep_pids:
                continue
            md5 = r.get('protein_md5', '').strip()
            k = (r.get('host_K') or '').strip()
            if md5 and k and k != 'N/A':
                md5_to_K[md5].add(normK(k))
                n_kept_rows += 1
    scope_label = ('pipeline_positive (the ESM-2 3B hybrid K-source training set)'
                   if RESTRICT_TO_PIPELINE_POSITIVE
                   else 'all training entries (broad association table)')
    print(f'  scope: {scope_label}')
    print(f'  {len(md5_to_K):,} training MD5s with K-labels (from {n_kept_rows:,} association rows)')

    print('[4/4] running drilldown for PHL and PBIP (all phages, not just misses)')
    phl_rows = drilldown_for_dataset(VAL_BASE, 'PhageHostLearn', val_to_hits, pid_to_md5, md5_to_K)
    pbip_rows = drilldown_for_dataset(VAL_BASE, 'PBIP', val_to_hits, pid_to_md5, md5_to_K)
    print(f'  PHL pairs at ≥{PIDENT_FLOOR:.0f}% id: {len(phl_rows)}')
    print(f'  PBIP pairs at ≥{PIDENT_FLOOR:.0f}% id: {len(pbip_rows)}')

    def summarize(rows, label):
        n = len(rows)
        if n == 0:
            return n, 0, 0
        n_dom = sum(1 for r in rows if r['matches_dom'])
        n_any = sum(1 for r in rows if r['matches_any'])
        print(f'  {label}: {n} pairs')
        print(f'    matches dominant K     : {n_dom} ({100*n_dom/n:.0f}%)')
        print(f'    matches any host K     : {n_any} ({100*n_any/n:.0f}%)')
        return n, n_dom, n_any

    phl_n, phl_dom, phl_any = summarize(phl_rows, 'PHL (all phages)')
    pbip_n, pbip_dom, pbip_any = summarize(pbip_rows, 'PBIP (all phages)')

    # ── Plot — side-by-side comparison ──
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = ['matches\ndominant K', 'matches\nany positive\nhost K', 'no match\nto any\nhost K']
    phl_pcts = [100 * phl_dom / max(phl_n, 1),
                100 * (phl_any - phl_dom) / max(phl_n, 1),
                100 * (phl_n - phl_any) / max(phl_n, 1)]
    pbip_pcts = [100 * pbip_dom / max(pbip_n, 1),
                 100 * (pbip_any - pbip_dom) / max(pbip_n, 1),
                 100 * (pbip_n - pbip_any) / max(pbip_n, 1)]

    width = 0.35
    x = np.arange(len(cats))
    bars_phl = ax.bar(x - width/2, phl_pcts, width, label=f'PHL (n={phl_n} pairs)',
                       color='#d7191c', edgecolor='white', linewidth=1.5)
    bars_pbip = ax.bar(x + width/2, pbip_pcts, width, label=f'PBIP (n={pbip_n} pairs)',
                        color='#2c7bb6', edgecolor='white', linewidth=1.5)
    for bars in (bars_phl, bars_pbip):
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                    f'{b.get_height():.0f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylabel('% of (RBP, training-neighbour) pairs')
    ax.set_ylim(0, max(max(phl_pcts), max(pbip_pcts)) * 1.25)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    scope_short = ('pipeline_positive training set'
                   if RESTRICT_TO_PIPELINE_POSITIVE else 'all training entries')
    ax.set_title(
        f'PBIP vs PHL — high-id (≥{PIDENT_FLOOR:.0f}%) training-neighbour K-label agreement\n'
        f'(all phages in each dataset; training pool: {scope_short})',
        fontsize=10.5, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT_SVG}')
    print(f'  wrote {OUT_PNG}')


if __name__ == '__main__':
    main()
