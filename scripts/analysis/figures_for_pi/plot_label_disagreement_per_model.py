"""Per-model PHL vs PBIP K-label agreement of high-id training neighbours.

The headline 2-hybrid uses different K- and O-source models, each trained
on a different filter of the underlying glycan-binders pool:

    K-source: sweep_posList_esm2_3b_mean_cl70  (--positive_list pipeline_positive.list)
    O-source: sweep_kmer_aa20_k4               (--tools DepoScope,PhageRBPdetect)

So "what does the model's training set look like" is two different answers.
This script computes the high-id training-neighbour K-label agreement
rate for both models, on both PHL and PBIP, in a single grouped-bar plot.

Each cell answers: of all (validation-RBP, training-neighbour) pairs at
≥80% identity within this model's actual training pool, what fraction
of training neighbours carry a K-label that matches ANY of the
validation phage's positive-host K-types?

Output:
  results/figures/pi_meeting_2026-05-05/02_phl_gap_diagnosis/
      fig_label_disagreement_per_model.{svg,png}
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
GB = CIPHER / 'data/training_data/metadata/glycan_binders_custom.tsv'
PP = CIPHER / 'data/training_data/metadata/pipeline_positive.list'

OUT_DIR = (CIPHER / 'results/figures/pi_meeting_2026-05-05/02_phl_gap_diagnosis')
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SVG = OUT_DIR / 'fig_label_disagreement_per_model.svg'
OUT_PNG = OUT_DIR / 'fig_label_disagreement_per_model.png'

PIDENT_FLOOR = 80.0


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


# ── Build the protein_id sets each model ACTUALLY trained on ──────────────

def kmer_model_pids():
    """sweep_kmer_aa20_k4: --tools DepoScope,PhageRBPdetect → keep proteins
    flagged by AT LEAST ONE of those tools in glycan_binders."""
    keep = set()
    with GB.open() as fh:
        r = csv.DictReader(fh, delimiter='\t')
        for row in r:
            try:
                if int(row.get('DepoScope', 0)) == 1 or int(row.get('PhageRBPdetect', 0)) == 1:
                    keep.add(row['protein_id'])
            except (TypeError, ValueError):
                continue
    return keep


def esm2_3b_model_pids():
    """sweep_posList_esm2_3b_mean_cl70: --positive_list pipeline_positive.list."""
    with PP.open() as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def build_md5_to_K(allowed_pids):
    """{md5: set(K-labels)} restricted to training rows with protein_id in allowed_pids."""
    md5_to_K = defaultdict(set)
    with HOST_MAP.open() as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r.get('protein_id', '').strip() not in allowed_pids:
                continue
            md5 = r.get('protein_md5', '').strip()
            k = (r.get('host_K') or '').strip()
            if md5 and k and k != 'N/A':
                md5_to_K[md5].add(normK(k))
    return md5_to_K


# ── Load val hits + per-dataset (phage → RBP md5s, K-set) ────────────────

def load_val_hits():
    """{val_md5: list[(train_md5, pident)] sorted desc by pident}."""
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


def load_dataset_phages(dataset_name):
    """Return ({phage: [protein_id, ...]}, {phage: {host K-types}})."""
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


# ── Drilldown: agreement rate per (model, dataset) ────────────────────────

def drilldown(val_hits, pid_to_md5, md5_to_K, dataset_name):
    """Return dict with n_pairs, n_dom_match (counted as any-match here for
    simplicity), n_any_match for the given dataset."""
    phage_proteins, K_set = load_dataset_phages(dataset_name)
    n_pairs = n_any = 0
    for ph, true_K in K_set.items():
        for pid in phage_proteins.get(ph, []):
            md5 = pid_to_md5.get(pid)
            if not md5:
                continue
            hits = val_hits.get(md5, [])
            if not hits:
                continue
            top = [(t, p) for t, p in hits if p >= PIDENT_FLOOR][:5]
            for t_md5, _ in top:
                t_K_set = md5_to_K.get(t_md5, set())
                if not t_K_set:
                    # Training neighbour exists in the model's training
                    # pool BUT has no K-label there → still counts as
                    # "in pool", since that's what we asked
                    n_pairs += 1
                    continue
                n_pairs += 1
                if t_K_set & true_K:
                    n_any += 1
    return {'n_pairs': n_pairs, 'n_any_match': n_any}


def main():
    print('[1/3] loading val hits + val pid→md5')
    val_hits = load_val_hits()
    pid_to_md5 = {pid: md5_str(seq) for pid, seq in parse_fasta(VAL_FAA)}
    print(f'  {len(val_hits)} val MD5s with hits, {len(pid_to_md5)} val proteins')

    print('[2/3] building per-model training sets')
    kmer_pids = kmer_model_pids()
    esm2_pids = esm2_3b_model_pids()
    print(f'  kmer model (DepoScope|PhageRBPdetect): {len(kmer_pids):,} protein_ids')
    print(f'  esm2_3b model (pipeline_positive):     {len(esm2_pids):,} protein_ids')
    overlap = kmer_pids & esm2_pids
    print(f'  overlap: {len(overlap):,}  (kmer-only={len(kmer_pids - esm2_pids):,}, '
          f'esm2-only={len(esm2_pids - kmer_pids):,})')

    print('  building md5→K maps from each pool')
    kmer_m2k = build_md5_to_K(kmer_pids)
    esm2_m2k = build_md5_to_K(esm2_pids)
    print(f'  kmer pool: {len(kmer_m2k):,} MD5s with K-labels')
    print(f'  esm2 pool: {len(esm2_m2k):,} MD5s with K-labels')

    print('[3/3] drilldown per (model, dataset)')
    results = {}
    for model_name, m2k in [('kmer (O-source)', kmer_m2k),
                              ('esm2_3b (K-source)', esm2_m2k)]:
        for ds in ['PhageHostLearn', 'PBIP']:
            stats = drilldown(val_hits, pid_to_md5, m2k, ds)
            pct = 100 * stats['n_any_match'] / max(1, stats['n_pairs'])
            results[(model_name, ds)] = (pct, stats['n_pairs'], stats['n_any_match'])
            print(f"  {model_name:<22s}  {ds:<14s}  "
                  f"any-match: {stats['n_any_match']}/{stats['n_pairs']} = {pct:.1f}%")

    # ── Plot — grouped bars: x = dataset, hue = model ─────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ds_order = ['PhageHostLearn', 'PBIP']
    model_order = ['kmer (O-source)', 'esm2_3b (K-source)']
    colors = {'kmer (O-source)': '#1b9e77', 'esm2_3b (K-source)': '#7570b3'}
    width = 0.35
    x = np.arange(len(ds_order))
    for i, model in enumerate(model_order):
        offsets = (i - 0.5) * width
        ys = [results[(model, ds)][0] for ds in ds_order]
        ns = [results[(model, ds)][1] for ds in ds_order]
        bars = ax.bar(x + offsets, ys, width, label=f'{model}', color=colors[model],
                      edgecolor='white', linewidth=1.5)
        for b, y, n in zip(bars, ys, ns):
            ax.text(b.get_x() + b.get_width() / 2, y + 1,
                    f'{y:.0f}%\n(n={n})',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(ds_order, fontsize=11)
    ax.set_ylabel('% of high-id (≥80%) training neighbours\nwith K-label matching ANY phage host K')
    ax.set_xlabel('Validation dataset')
    ax.set_ylim(0, max(r[0] for r in results.values()) * 1.4)
    ax.legend(loc='upper left', fontsize=9, title='Training pool used')
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(
        f'Same story across both head models?\n'
        f'High-id (≥{PIDENT_FLOOR:.0f}%) training-neighbour K-label agreement, by model\'s actual training pool',
        fontsize=10.5, fontweight='bold')

    fig.tight_layout()
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {OUT_SVG}')
    print(f'  wrote {OUT_PNG}')


if __name__ == '__main__':
    main()
