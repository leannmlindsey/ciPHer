"""HR@k=1..20 for PHL and PBIP under three denominator framings, for one or
more cipher runs. Addresses agent 5's 2026-04-30 ask:

  "Re-evaluate the top PHL models with the 27 PHL phages and their 77
   positive pairs excluded; report HR@1, HR@5, MRR vs the full denominator."

Three denominators considered:

  1. fixed_full     — n_phage = 127 PHL / 104 PBIP. **HEADLINE per
                      project policy** (memory: feedback_strict_denominator).
                      Every phage with a positive interaction in the
                      interaction matrix counts; phages the model can't
                      score for any reason count as always-miss.
  2. genome_corrected — fixed_full minus phages with NO genome anywhere
                      on disk (agent 5's "sidecar" denominator for advisor
                      evidence). PHL: 127 − 27 = 100. PBIP: 104 − 1 = 103.
                      Excludes only the defensible exception (zero possible
                      input surface), keeping every other miss in the denom.
  3. cipher_old_buggy — n_phage = n_strict_phage from
                      per_head_strict_eval.json (PHL=99, PBIP=101). The
                      harvest CSV's *_OR_phage2host_anyhit_HR<k> columns
                      currently use this; it violates fixed-denominator
                      policy because per_head_strict_eval.py:150 drops
                      every phage with no annotated RBP in cipher's
                      pipeline_positive, including phages that DO have
                      RBPs in other on-disk sources (S13c on PHL, 2 on
                      PBIP). Reported only to make the bug visible — do
                      not quote.

The script reads each run's `per_head_strict_eval.json` to recover the
numerator (number of phages with rank ≤ k) and then divides by each of
the three denominators. Output: a table to stdout + a 2-panel figure
(PHL, PBIP) showing HR@k under the three denominators, one curve set
per run.

Output:
  results/figures/recall_at_k_denominator_corrections.svg/.png
  results/analysis/recall_at_k_denominator_corrections.tsv

Usage:
  python scripts/analysis/recall_under_denominator_corrections.py
  python scripts/analysis/recall_under_denominator_corrections.py \
      --runs sweep_prott5_mean_cl70 highconf_pipeline_K_prott5_mean \
      --mode or
"""

import argparse
import csv
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


HARVEST_CSV = 'results/experiment_log.csv'
PHL_NO_GENOME = ('/Users/leannmlindsey/WORK/CLAUDE_PHI_alignment/analyses/'
                 '27_advisor_html_report/output/lists/'
                 'phl_phages_no_genome_annotation.txt')
PBIP_NO_GENOME = ('/Users/leannmlindsey/WORK/CLAUDE_PHI_alignment/analyses/'
                  '27_advisor_html_report/output/lists/'
                  'pbip_phages_no_genome_annotation.txt')

# Total phages with positive interactions per dataset (the "naive full"
# denominator). These come from the structural-ceiling table in
# notes/findings/2026-04-28_phl_structural_ceiling_missing_rbps.md.
NAIVE_FULL = {'PhageHostLearn': 127, 'PBIP': 104,
              'CHEN': 3, 'GORODNICHIV': 3, 'UCSD': 11}

DEFAULT_RUNS = [
    'sweep_prott5_mean_cl70',
    'sweep_esm2_650m_mean_cl70',
    'concat_prott5_mean+kmer_li10_k5',
    'highconf_pipeline_K_prott5_mean',
]
DEFAULT_MODE = 'or'  # or | k_only | o_only | merged

OUT_SVG = 'results/figures/recall_at_k_denominator_corrections.svg'
OUT_TSV = 'results/analysis/recall_at_k_denominator_corrections.tsv'


def load_phages_to_exclude(path):
    if not os.path.exists(path):
        print(f'WARN: list not found at {path} — using empty set')
        return set()
    with open(path) as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def find_eval_json(run_name):
    """Search for per_head_strict_eval.json in main + sibling worktrees."""
    candidates = []
    here = Path('.').resolve()
    siblings = sorted(here.parent.glob('cipher*'))
    for root in [here] + [s for s in siblings if s != here]:
        for arch in ('attention_mlp', 'light_attention',
                      'light_attention_binary', 'contrastive_encoder'):
            p = root / 'experiments' / arch / run_name / 'results' / \
                'per_head_strict_eval.json'
            if p.exists():
                candidates.append(p)
    return candidates[0] if candidates else None


def numerators_per_k(eval_json, dataset, mode):
    """Return ({k: n_hits_phage_anyhit}, n_strict_phage) for dataset+mode.

    Reconstructs numerators from HR@k * n_strict_phage so we can re-divide.
    The headline column we compute is "phage-level any-hit" (matches
    *_OR_phage2host_anyhit_HR<k> in the harvest CSV).
    """
    with open(eval_json) as fh:
        d = json.load(fh)
    blk = d.get(dataset)
    if not blk:
        return None, None
    n_phage = blk['n_strict_phage']
    mode_blk = blk.get(mode)
    if not mode_blk:
        return None, None
    # `hr_at_k_any_hit` is the phage-level any-hit (line 259 of
    # per_head_strict_eval.py), denominator = n_strict_phage.
    hrk = mode_blk.get('hr_at_k_any_hit', {})
    nums = {}
    for k_str, v in hrk.items():
        k = int(k_str)
        nums[k] = round(v * n_phage)
    return nums, n_phage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', default=DEFAULT_RUNS)
    ap.add_argument('--mode', default=DEFAULT_MODE,
                    choices=('or', 'k_only', 'o_only', 'merged'))
    ap.add_argument('--datasets', nargs='+',
                    default=['PhageHostLearn', 'PBIP'])
    ap.add_argument('--out-svg', default=OUT_SVG)
    ap.add_argument('--out-tsv', default=OUT_TSV)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_svg) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(args.out_tsv) or '.', exist_ok=True)

    phl_excl = load_phages_to_exclude(PHL_NO_GENOME)
    pbip_excl = load_phages_to_exclude(PBIP_NO_GENOME)
    print(f'PHL no-genome list: {len(phl_excl)} phages')
    print(f'PBIP no-genome list: {len(pbip_excl)} phages')

    # Effective "genome_corrected" denominator per dataset.
    genome_denom = {
        'PhageHostLearn': NAIVE_FULL['PhageHostLearn'] - len(phl_excl),
        'PBIP':           NAIVE_FULL['PBIP'] - len(pbip_excl),
    }

    rows_out = []
    # data[run][dataset][denom_key] = {k: hr}
    data = {}

    for run in args.runs:
        ej = find_eval_json(run)
        if ej is None:
            print(f'WARN: no per_head_strict_eval.json for {run}; skipping')
            continue
        data[run] = {}
        for ds in args.datasets:
            nums, n_strict = numerators_per_k(ej, ds, args.mode)
            if nums is None:
                continue
            naive = NAIVE_FULL.get(ds)
            ag5 = genome_denom.get(ds, n_strict)
            data[run][ds] = {
                'fixed_full':       {k: v / naive    for k, v in nums.items()},
                'genome_corrected': {k: v / ag5      for k, v in nums.items()},
                'cipher_old_buggy': {k: v / n_strict for k, v in nums.items()},
                'n_naive':          naive,
                'n_genome':         ag5,
                'n_strict':         n_strict,
                'hits':             nums,
            }
            for denom_label, denom_n in (('fixed_full',       naive),
                                          ('genome_corrected', ag5),
                                          ('cipher_old_buggy', n_strict)):
                for k in (1, 5, 10, 20):
                    rows_out.append({
                        'run':       run,
                        'dataset':   ds,
                        'mode':      args.mode,
                        'denom':     denom_label,
                        'n_phages':  denom_n,
                        'k':         k,
                        'hits':      nums.get(k),
                        'hr_at_k':   round(nums.get(k, 0) / denom_n, 4),
                    })

    # ── TSV summary ──
    with open(args.out_tsv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()),
                           delimiter='\t')
        w.writeheader()
        for r in rows_out:
            w.writerow(r)
    print(f'Wrote {args.out_tsv}')

    # ── Print headline table ──
    print()
    fmt = '  {:<32}  {:<14}  {:>10}  {:>5}  ' + '  '.join(['{:>7}'] * 4)
    print('Numerator (hits) and HR@k under each denominator framing.')
    for ds in args.datasets:
        print(f'\n{ds}:')
        print(fmt.format('run', 'denom', 'n', 'hits',
                          'HR@1', 'HR@5', 'HR@10', 'HR@20'))
        for run in args.runs:
            if run not in data or ds not in data[run]:
                continue
            d = data[run][ds]
            for denom_label in ('fixed_full', 'genome_corrected',
                                  'cipher_old_buggy'):
                n_key = {'fixed_full': 'n_naive',
                         'genome_corrected': 'n_genome',
                         'cipher_old_buggy': 'n_strict'}[denom_label]
                print(fmt.format(
                    run[:32], denom_label, d[n_key], d['hits'].get(1, 0),
                    *[f"{d[denom_label][k]:.3f}" if k in d[denom_label]
                      else '   --' for k in (1, 5, 10, 20)]))

    # ── Figure: 2 panels (PHL, PBIP) × 3 denoms × N runs ──
    fig, axes = plt.subplots(1, len(args.datasets), figsize=(7*len(args.datasets), 6.0),
                              squeeze=False)
    ks = list(range(1, 21))

    # Distinct line style per denominator, distinct color per run.
    # fixed_full = headline (solid, thick); the others are sidecars.
    denom_style = {
        'fixed_full':       ('-',  2.4, 'fixed full denom (project policy headline)'),
        'genome_corrected': ('--', 2.0, 'genome-corrected (sidecar; exclude no-genome only)'),
        'cipher_old_buggy': (':',  1.6, 'cipher_old_buggy (DO NOT QUOTE)'),
    }

    cmap = plt.get_cmap('tab10')
    for j, ds in enumerate(args.datasets):
        ax = axes[0, j]
        for i, run in enumerate(args.runs):
            if run not in data or ds not in data[run]:
                continue
            color = cmap(i % cmap.N)
            d = data[run][ds]
            for denom_label, (ls, lw, _) in denom_style.items():
                ys = [d[denom_label].get(k) for k in ks]
                if all(v is None for v in ys):
                    continue
                # Use the FIXED FULL curve as the canonical legend entry
                # for each run (project policy).
                first_for_run = (denom_label == 'fixed_full')
                ax.plot(ks, ys, color=color, lw=lw, linestyle=ls,
                        marker='o' if first_for_run else None,
                        markersize=3.5,
                        label=(run if first_for_run else None))

        n_naive = NAIVE_FULL[ds]
        n_strict = next(iter(data.values()))[ds]['n_strict'] \
            if data and ds in next(iter(data.values())) else '?'
        n_geno = genome_denom.get(ds, '?')
        ax.set_title(f'{ds}\n'
                     f'fixed_full={n_naive}  genome_corrected={n_geno}  '
                     f'cipher_old_buggy={n_strict}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('k')
        if j == 0:
            ax.set_ylabel(f'Recall@k (phage-level any-hit, mode={args.mode})')
        ax.set_xlim(0.5, 20.5)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9, title='run')

    # Single shared bottom legend explaining the linestyles.
    style_handles = []
    for ls, lw, lab in [denom_style[k] for k in
                          ('fixed_full', 'cipher_old_buggy', 'genome_corrected')]:
        style_handles.append(plt.Line2D([0], [0], color='black', lw=lw,
                                         linestyle=ls, label=lab))
    fig.legend(handles=style_handles, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02), framealpha=0.95,
               title='denominator framing')

    fig.suptitle('PHL/PBIP recall@k under three denominator framings\n'
                 'numerator unchanged; only the divisor differs',
                 fontsize=11, fontweight='bold', y=1.00)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(args.out_svg, format='svg', bbox_inches='tight')
    fig.savefig(args.out_svg.replace('.svg', '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'\nWrote {args.out_svg}')


if __name__ == '__main__':
    main()
