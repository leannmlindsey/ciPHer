"""K-head vs O-head per-pair rank scatter on a validation dataset.

For every positive (phage, host) pair, computes the rank of the true host
under cipher's K-only head and O-only head separately (by masking the
unused head's output to -inf). Renders a scatter:

    x = K-head rank of true host (1 = best, max = candidate-pool size)
    y = O-head rank of true host

Color coding (default top-k threshold = 5):
    green   — both heads put the host in top-k    (both succeed)
    blue    — K rank ≤ k, O rank > k              (K wins, O fails)
    red     — O rank ≤ k, K rank > k              (O wins, K fails)
    grey    — both > k                            (both fail or partial)

A vertical "pin line" at x = max indicates phages whose true K-type is
out-of-vocabulary in the model's K-head label space. Likewise for y.

Caches per-pair ranks to a TSV alongside the figure so re-plotting is cheap.

Output:
    results/figures/k_o_head_scatter_<run_name>_<dataset>.{svg,png}
    results/analysis/k_o_head_ranks_<run_name>_<dataset>.tsv

Usage (concat run on Delta):
    python scripts/analysis/plot_k_o_head_scatter.py \
        --experiment-dir experiments/attention_mlp/concat_prott5_mean+kmer_li10_k5 \
        --val-fasta data/validation_data/metadata/validation_rbps_all.faa \
        --val-embedding-file /path/to/validation_prott5_mean.npz \
        --val-embedding-file-2 /path/to/validation_kmer_li10_k5.npz \
        --val-datasets-dir data/validation_data/HOST_RANGE \
        --dataset PhageHostLearn

Re-plot from cached TSV (no model load):
    python scripts/analysis/plot_k_o_head_scatter.py \
        --per-pair-tsv results/analysis/k_o_head_ranks_<run>_<ds>.tsv \
        --label "concat_prott5+kmer_li10_k5"
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    # Compute mode (loads predictor)
    p.add_argument('--experiment-dir',
                   help='cipher experiment dir (model weights). Required to compute.')
    p.add_argument('--val-fasta', help='validation_rbps_all.faa')
    p.add_argument('--val-embedding-file',
                   help='primary validation NPZ (md5 → embedding)')
    p.add_argument('--val-embedding-file-2', default=None,
                   help='second validation NPZ for concat models (optional)')
    p.add_argument('--val-datasets-dir',
                   help='dir containing <DATASET>/metadata/interaction_matrix.tsv')
    p.add_argument('--dataset', default='PhageHostLearn',
                   choices=('CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn'))
    # Plot-only mode (two TSV schemas accepted)
    p.add_argument('--per-pair-tsv', default=None,
                   help='use a previously computed per-pair TSV (skip predictor)')
    p.add_argument('--per-phage-tsv', default=None,
                   help='use a per-phage any-hit TSV from the hybrid pipeline '
                        '(columns: dataset, phage_id, k_only_rank, o_only_rank, ...). '
                        'One dot per phage (best rank across positives), not per pair.')
    p.add_argument('--label', default=None,
                   help='legend / title label (defaults to run_name from path)')
    # Plot configuration
    p.add_argument('--top-k', type=int, default=5,
                   help='rank threshold for the "correct" colour bucket')
    p.add_argument('--annotate', nargs='*', default=[],
                   help='phage IDs to annotate on the figure')
    p.add_argument('--out-svg', default=None)
    p.add_argument('--out-tsv', default=None)
    p.add_argument('--cipher-src', default=None,
                   help='path to cipher src/ if not on PYTHONPATH')
    return p.parse_args()


# ───────────────────── Compute (predictor-driven) ─────────────────────

def _mask_head(orig, mode):
    """Returns a wrapper of predict_protein that nulls one head's output.
    mode in {'k_only','o_only'}. Same convention as per_head_strict_eval."""
    def wrapped(emb):
        out = orig(emb)
        if mode == 'k_only':
            out = {**out, 'o_probs': {k: float('-inf') for k in out.get('o_probs', {})}}
        elif mode == 'o_only':
            out = {**out, 'k_probs': {k: float('-inf') for k in out.get('k_probs', {})}}
        return out
    return wrapped


def compute_per_pair(args):
    if args.cipher_src and args.cipher_src not in sys.path:
        sys.path.insert(0, args.cipher_src)
    from cipher.evaluation.ranking import rank_hosts
    from cipher.data.interactions import load_interaction_pairs
    from cipher.data.proteins import load_fasta_md5
    from cipher.data.embeddings import load_embeddings, load_embeddings_concat

    # Load predictor (cipher's get_predictor handles concat dual-embedding)
    sys.path.insert(0, str(Path(args.experiment_dir).resolve().parents[1]
                           / Path(args.experiment_dir).name))
    # Each model package exposes predict.get_predictor(run_dir)
    import importlib.util
    pred_path = (Path(args.experiment_dir).resolve().parents[1]
                 / 'attention_mlp' / 'predict.py')  # default; may be light_attention etc.
    arch = Path(args.experiment_dir).resolve().parents[0].name
    cipher_root = Path(args.experiment_dir).resolve().parents[2]
    spec_path = cipher_root / 'models' / arch / 'predict.py'
    spec = importlib.util.spec_from_file_location('_pred', spec_path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    predictor = mod.get_predictor(str(Path(args.experiment_dir).resolve()))
    original_predict = predictor.predict_protein

    # Load val embeddings
    fasta_md5 = load_fasta_md5(args.val_fasta)  # {protein_id: md5}
    if args.val_embedding_file_2:
        emb_dict = load_embeddings_concat(
            args.val_embedding_file, args.val_embedding_file_2,
            md5_filter=set(fasta_md5.values()))
    else:
        emb_dict = load_embeddings(args.val_embedding_file,
                                    md5_filter=set(fasta_md5.values()))

    # Load PHL/PBIP/etc. interaction data
    ds_dir = os.path.join(args.val_datasets_dir, args.dataset)
    pairs = load_interaction_pairs(ds_dir)
    pm_path = os.path.join(ds_dir, 'metadata', 'phage_protein_mapping.csv')
    from cipher.data.interactions import load_phage_protein_mapping
    phage_protein_map = load_phage_protein_mapping(pm_path)

    interactions = defaultdict(dict)
    serotypes = {}
    for p in pairs:
        interactions[p['phage_id']][p['host_id']] = p['label']
        serotypes[p['host_id']] = {'K': p['host_K'], 'O': p['host_O']}

    pos_pairs = []
    for phage in sorted(interactions):
        for h, lbl in interactions[phage].items():
            if lbl == 1:
                pos_pairs.append((phage, h))

    print(f'{args.dataset}: {len(pos_pairs)} positive pairs '
          f'over {len({p for p,_ in pos_pairs})} phages, '
          f'pool ≈ {len(serotypes)} candidate hosts')

    # Compute K-only and O-only ranks per phage (cache by phage)
    from cipher.evaluation.ranking import _ranks_with_ties as _rank_ties
    pool_size = len(serotypes)

    rows = []
    for mode_label in ('k_only', 'o_only'):
        predictor.predict_protein = _mask_head(original_predict, mode_label)
        per_phage_h2r = {}
        ranks_for_mode = {}
        for phage, host in pos_pairs:
            if phage not in per_phage_h2r:
                proteins = phage_protein_map.get(phage, set())
                candidates = list(interactions[phage].keys())
                ranked = rank_hosts(predictor, proteins, candidates,
                                    serotypes, emb_dict, fasta_md5)
                per_phage_h2r[phage] = (
                    _rank_ties(ranked, tie_method='competition')
                    if ranked else {})
            ranks_for_mode[(phage, host)] = per_phage_h2r[phage].get(host)
        if mode_label == 'k_only':
            k_ranks = ranks_for_mode
        else:
            o_ranks = ranks_for_mode

    for phage, host in pos_pairs:
        rows.append({
            'phage_id': phage,
            'host_id': host,
            'host_K': serotypes[host]['K'] or '',
            'host_O': serotypes[host]['O'] or '',
            'k_only_rank': k_ranks[(phage, host)] if k_ranks[(phage, host)] is not None else pool_size,
            'o_only_rank': o_ranks[(phage, host)] if o_ranks[(phage, host)] is not None else pool_size,
            'pool_size': pool_size,
        })
    return rows


# ────────────────────────── Plot ──────────────────────────

def render(rows, args):
    label = args.label or (
        Path(args.experiment_dir).name if args.experiment_dir else 'cipher')
    pool = max(int(r['pool_size']) for r in rows) if rows else 220

    K = args.top_k
    buckets = {'both': [], 'k_wins': [], 'o_wins': [], 'fail': []}
    by_phage = {}
    for r in rows:
        kr = int(r['k_only_rank']); orr = int(r['o_only_rank'])
        by_phage.setdefault(r['phage_id'], []).append(r)
        if kr <= K and orr <= K:    buckets['both'].append((kr, orr, r))
        elif kr <= K and orr > K:   buckets['k_wins'].append((kr, orr, r))
        elif orr <= K and kr > K:   buckets['o_wins'].append((kr, orr, r))
        else:                        buckets['fail'].append((kr, orr, r))

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    style = [
        ('fail',    'lightgrey', 'both fail or partial', 0.55, 14),
        ('both',    '#2ca02c',   f'both heads correct (rank ≤ {K})', 0.85, 24),
        ('k_wins',  '#1f77b4',   f'K wins, O fails',                  0.85, 22),
        ('o_wins',  '#d62728',   f'O wins, K fails',                  0.85, 22),
    ]
    for key, color, lab, alpha, sz in style:
        pts = buckets[key]
        if not pts: continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.scatter(xs, ys, s=sz, c=color, alpha=alpha,
                   label=f'{lab}  (n={len(pts)})',
                   edgecolors='none')

    # Pin-line annotations: how many points are pinned at pool_size
    n_kpin = sum(1 for r in rows if int(r['k_only_rank']) >= pool)
    n_opin = sum(1 for r in rows if int(r['o_only_rank']) >= pool)
    if n_kpin:
        ax.text(pool, pool*0.96, f'{n_kpin} K-OOV',
                ha='right', va='top', fontsize=9, color='#555',
                bbox=dict(boxstyle='round,pad=0.25', fc='#fff8e1', ec='#bbb', alpha=0.9))
    if n_opin:
        ax.text(pool*0.05, pool, f'{n_opin} O-OOV',
                ha='left', va='top', fontsize=9, color='#555',
                bbox=dict(boxstyle='round,pad=0.25', fc='#fff8e1', ec='#bbb', alpha=0.9))

    # Optional phage annotations
    for ph_id in args.annotate:
        if ph_id in by_phage:
            r = by_phage[ph_id][0]
            kr = int(r['k_only_rank']); orr = int(r['o_only_rank'])
            ax.annotate(f'{ph_id}\nK={r["host_K"]} O={r["host_O"]}',
                        xy=(kr, orr), xytext=(15, 15),
                        textcoords='offset points', fontsize=8,
                        arrowprops=dict(arrowstyle='-', color='#555', lw=0.6),
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#888', alpha=0.95))

    ax.set_xlim(-pool*0.02, pool*1.04)
    ax.set_ylim(-pool*0.02, pool*1.04)
    ax.set_xlabel(f'K-head rank of true host  (1 = best, ≥{pool} = bottom-of-pool / OOV)')
    ax.set_ylabel(f'O-head rank of true host  (1 = best, ≥{pool} = bottom-of-pool / OOV)')
    ax.set_title(f'K-head vs O-head per-pair rank on {args.dataset}\n'
                 f'positive (phage, host) pairs; {label}',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

    out_svg = args.out_svg or (
        f'results/figures/k_o_head_scatter_{label}_{args.dataset}.svg'
        .replace('+', 'plus'))
    os.makedirs(os.path.dirname(out_svg) or '.', exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    fig.savefig(out_svg.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_svg}')

    # Print bucket summary
    print()
    print(f'{args.dataset} {label}: {len(rows)} positive pairs')
    for key, _, lab, *_ in style:
        n = len(buckets[key])
        pct = 100.0 * n / max(len(rows), 1)
        print(f'  {lab:<40s}  n={n:>4d}  ({pct:5.1f}%)')


def write_tsv(rows, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), delimiter='\t')
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f'Wrote {path}')


def read_tsv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter='\t'))


def _adapt_per_phage(rows, dataset_filter=None, pool_default=220):
    """Convert per-phage TSV (dataset, phage_id, k_only_rank, o_only_rank, ...)
    to the per-pair schema expected by render(). One dot per phage."""
    out = []
    for r in rows:
        if dataset_filter and r.get('dataset') != dataset_filter:
            continue
        kr = r.get('k_only_rank') or ''
        orr = r.get('o_only_rank') or ''
        try: kr = int(kr) if kr.strip() else pool_default
        except (ValueError, AttributeError): kr = pool_default
        try: orr = int(orr) if orr.strip() else pool_default
        except (ValueError, AttributeError): orr = pool_default
        out.append({
            'phage_id': r['phage_id'],
            'host_id': '',
            'host_K': r.get('host_K', ''),
            'host_O': r.get('host_O', ''),
            'k_only_rank': kr,
            'o_only_rank': orr,
            'pool_size': pool_default,
        })
    return out


def main():
    args = parse_args()
    if args.per_pair_tsv:
        rows = read_tsv(args.per_pair_tsv)
        if not args.label:
            args.label = (Path(args.per_pair_tsv).stem
                          .replace('k_o_head_ranks_', '')
                          .rsplit('_', 1)[0])
        render(rows, args)
        return
    if args.per_phage_tsv:
        raw = read_tsv(args.per_phage_tsv)
        rows = _adapt_per_phage(raw, dataset_filter=args.dataset)
        if not args.label:
            args.label = (Path(args.per_phage_tsv).stem
                          .replace('per_phage_', ''))
        render(rows, args)
        return
    # Compute mode requires the model
    for required in ('experiment_dir', 'val_fasta', 'val_embedding_file',
                      'val_datasets_dir'):
        if not getattr(args, required):
            sys.exit(f'ERROR: --{required.replace("_", "-")} is required '
                     f'when not using --per-pair-tsv')
    rows = compute_per_pair(args)
    label = (Path(args.experiment_dir).name).replace('+', 'plus')
    out_tsv = args.out_tsv or f'results/analysis/k_o_head_ranks_{label}_{args.dataset}.tsv'
    write_tsv(rows, out_tsv)
    if not args.label:
        args.label = Path(args.experiment_dir).name
    render(rows, args)


if __name__ == '__main__':
    main()
