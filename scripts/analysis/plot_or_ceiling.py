"""Plot K-only, O-only, K∪O (perfect-merge ceiling), and actual merged
HR@k=1..20 for one (or several) old_style_eval.json file(s).

Designed to answer: "if we combine K predictions and O predictions
optimally, where is the ceiling, and how much is the actual merge
leaving on the table?"

Reads the JSON(s) emitted by scripts/analysis/old_style_eval.py.
Uses the strict-denominator HR@k families:
  - hr_at_k        actual merged class-rank
  - k_hr_at_k      K-only class-rank
  - o_hr_at_k      O-only class-rank
  - or_hr_at_k     K∪O ceiling

Default dataset: PhageHostLearn. Pass --dataset to switch.

Usage:
    # Local dual-head (highconf K + LAPTOP O):
    python scripts/analysis/plot_or_ceiling.py \\
        results/dual_head_old_style/highconf_pipeline_K_prott5_mean_x_repro_old_v3_in_cipher_LAPTOP_20260425_235817/old_style_eval.json \\
        --label 'highconf K + LAPTOP O'

    # Single experiment:
    python scripts/analysis/plot_or_ceiling.py \\
        experiments/attention_mlp/repro_old_v3_in_cipher_LAPTOP_20260425_235817/results_old_style/old_style_eval.json \\
        --label 'LAPTOP repro'
"""

import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(json_path, dataset):
    with open(json_path) as f:
        d = json.load(f)
    if dataset not in d:
        raise SystemExit(f'{dataset!r} not in {json_path} (have: {list(d.keys())})')
    r = d[dataset]
    n = r.get('n_pairs')
    def curve(field):
        hrk = r.get(field, {})
        return [hrk.get(str(k), hrk.get(k, 0)) for k in range(1, 21)]
    return {
        'n': n,
        'merged': curve('hr_at_k'),
        'k': curve('k_hr_at_k'),
        'o': curve('o_hr_at_k'),
        'or': curve('or_hr_at_k'),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('json_path')
    p.add_argument('--dataset', default='PhageHostLearn')
    p.add_argument('--label', default=None,
                   help='Subtitle text describing K+O config')
    p.add_argument('--out', default=None,
                   help='Default: results/figures/or_ceiling_<dataset>.svg')
    args = p.parse_args()

    data = load(args.json_path, args.dataset)
    ks = list(range(1, 21))

    label = args.label or os.path.basename(os.path.dirname(args.json_path))

    fig, ax = plt.subplots(figsize=(8, 5.2))

    # K-only and O-only — the per-head ceilings
    ax.plot(ks, data['k'], color='#1f77b4', lw=2,
            marker='o', markersize=4,
            label=f'K-only  (HR@1={data["k"][0]:.3f})')
    ax.plot(ks, data['o'], color='#2ca02c', lw=2,
            marker='s', markersize=4,
            label=f'O-only  (HR@1={data["o"][0]:.3f})')
    # OR ceiling — the perfect-merge upper bound
    ax.plot(ks, data['or'], color='#d62728', lw=2.5,
            marker='^', markersize=4.5,
            label=f'K ∪ O  (HR@1={data["or"][0]:.3f})  ← perfect-merge ceiling')
    # Actual merge — what raw probability merging gives
    ax.plot(ks, data['merged'], color='#7f7f7f', lw=1.8,
            marker='x', markersize=4.5, linestyle='--',
            label=f'actual merge  (HR@1={data["merged"][0]:.3f})  ← raw-prob sort')

    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0, 1.02)
    ax.set_xticks([1, 2, 3, 5, 10, 15, 20])
    ax.set_xlabel('k')
    ax.set_ylabel('HR@k  (strict denominator)')
    ax.set_title(f'{args.dataset}: K + O combination ceiling vs actual merge\n'
                 f'{label}  (n={data["n"]})',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)

    fig.tight_layout()
    out = args.out or f'results/figures/or_ceiling_{args.dataset}.svg'
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, format='svg', bbox_inches='tight')
    fig.savefig(out.replace('.svg', '.png'), format='png',
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out}')
    print(f'      {out.replace(".svg", ".png")}')

    # Concise stdout summary
    print()
    print(f'  HR@k    K-only   O-only   K∪O     merged   merge_tax')
    print(f'  ----   ------   ------   -----   ------   ---------')
    for k in (1, 3, 5, 10, 20):
        kk = k - 1
        tax = data['or'][kk] - data['merged'][kk]
        print(f'  HR@{k:<2}  {data["k"][kk]:>6.3f}   {data["o"][kk]:>6.3f}   '
              f'{data["or"][kk]:>5.3f}   {data["merged"][kk]:>6.3f}   {tax:>+9.3f}')


if __name__ == '__main__':
    main()
