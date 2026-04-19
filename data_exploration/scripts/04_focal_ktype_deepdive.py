"""Per-MD5 specificity profile for one focal K-type.

Default focal K is K47 — well-represented in training and PHL but typically
underperforming. Override via --k-type CLI arg.

For each training MD5 with any K47 association:
- How many distinct K-types is it associated with?
- What's the K47 fraction of its observations?
- Is K47 its primary K-type?
- What does the model predict for it?

Reveals whether K47 underperformance is due to:
- Label noise (K47 proteins also map to many other K-types)
- Promiscuity (K47 proteins genuinely bind multiple K-types)
- Pure model failure (K47 labels are clean but model can't learn)

Outputs:
    data_exploration/output/focal_{Ktype}_md5_specificity.csv
    + summary block printed to stdout and saved as .txt
"""

import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
from _common import (
    OUTPUT_DIR, best_experiment_dir,
    load_training_data, load_raw_associations,
    load_predictor_for_inference, load_test_embeddings,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k-type', default='K47',
                        help='K-type to drill into (default: K47)')
    parser.add_argument('--top-n-other', type=int, default=3,
                        help='How many top other K-types to list per MD5')
    args = parser.parse_args()

    focal_k = args.k_type
    print(f'Focal K-type: {focal_k}')

    exp_dir = best_experiment_dir()
    print(f'Using experiment: {os.path.basename(exp_dir)}')

    print('Loading training data...')
    td, splits_k, _ = load_training_data(exp_dir)
    test_md5s = set(splits_k.get('test', []))

    print('Re-building raw per-MD5 K-type associations...')
    md5_k_counts, _ = load_raw_associations(exp_dir)
    print(f'  {len(md5_k_counts)} MD5s with K associations')

    # Filter to MD5s with any focal_k association
    focal_md5s = {m: c for m, c in md5_k_counts.items() if focal_k in c}
    print(f'  {len(focal_md5s)} MD5s have any {focal_k} association')

    if not focal_md5s:
        print(f'No MD5s with {focal_k} found. Try a different K-type.')
        return

    print('Loading predictor + embeddings for these MD5s...')
    predictor = load_predictor_for_inference(exp_dir)
    from cipher.data.embeddings import load_embeddings
    import yaml
    with open(os.path.join(exp_dir, 'config.yaml')) as f:
        emb_file = yaml.safe_load(f).get('data', {}).get(
            'embedding_file',
            'data/training_data/embeddings/esm2_650m_md5.npz')
    if not os.path.isabs(emb_file):
        from _common import REPO_ROOT
        emb_file = os.path.join(REPO_ROOT, emb_file)
    emb_dict = load_embeddings(emb_file, md5_filter=set(focal_md5s.keys()))
    print(f'  loaded {len(emb_dict)} embeddings')

    # Build per-MD5 rows
    rows = []
    n_with_emb = 0
    for md5, k_counter in sorted(focal_md5s.items()):
        total = sum(k_counter.values())
        focal_count = k_counter[focal_k]
        primary_k, primary_count = k_counter.most_common(1)[0]

        # Top-N other K-types (excluding the focal one if not primary)
        top_n = k_counter.most_common(args.top_n_other)
        top_n_str = '; '.join(f'{c}:{n}' for c, n in top_n)

        # Model prediction
        pred_k = ''
        pred_prob = ''
        if md5 in emb_dict:
            n_with_emb += 1
            preds = predictor.predict_protein(emb_dict[md5])
            k_probs = preds['k_probs']
            sorted_k = sorted(k_probs.items(), key=lambda x: -x[1])
            pred_k = sorted_k[0][0]
            pred_prob = f'{sorted_k[0][1]:.4f}'

        rows.append({
            'md5': md5,
            'n_distinct_k_types': len(k_counter),
            'total_observations': total,
            'primary_k': primary_k,
            'primary_k_count': primary_count,
            f'{focal_k}_count': focal_count,
            f'{focal_k}_fraction': f'{focal_count / total:.3f}',
            f'is_{focal_k}_primary': primary_k == focal_k,
            f'top_{args.top_n_other}_k_types': top_n_str,
            'model_predicted_k': pred_k,
            'model_predicted_prob': pred_prob,
            'is_in_test_split': md5 in test_md5s,
        })

    # Sort by focal fraction descending
    rows.sort(key=lambda r: -float(r[f'{focal_k}_fraction']))

    out_csv = os.path.join(OUTPUT_DIR, f'focal_{focal_k}_md5_specificity.csv')
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f'\nSaved: {out_csv} ({len(rows)} rows, {n_with_emb} with embeddings)')

    # Summary block
    n_total = len(rows)
    n_focal_primary = sum(1 for r in rows if r[f'is_{focal_k}_primary'])
    fractions = [float(r[f'{focal_k}_fraction']) for r in rows]
    mean_focal_frac = sum(fractions) / len(fractions)

    n_secondary = sum(1 for r in rows if not r[f'is_{focal_k}_primary']
                      and float(r[f'{focal_k}_fraction']) >= 0.25)
    n_minor = n_total - n_focal_primary - n_secondary

    n_focal_primary_correct = sum(
        1 for r in rows
        if r[f'is_{focal_k}_primary'] and r['model_predicted_k'] == focal_k)
    n_focal_primary_with_pred = sum(
        1 for r in rows
        if r[f'is_{focal_k}_primary'] and r['model_predicted_k'])

    distinct_k_dist = Counter(r['n_distinct_k_types'] for r in rows)

    summary_lines = [
        '=' * 70,
        f'FOCAL K-TYPE DEEP-DIVE: {focal_k}',
        '=' * 70,
        '',
        f'Experiment: {os.path.basename(exp_dir)}',
        f'Total MD5s with any {focal_k} association: {n_total}',
        '',
        f'  {focal_k} is primary K-type:                  {n_focal_primary}  ({100*n_focal_primary/n_total:.1f}%)',
        f'  {focal_k} is secondary (>=25% of obs):        {n_secondary}  ({100*n_secondary/n_total:.1f}%)',
        f'  {focal_k} is minor (<25% of obs):             {n_minor}  ({100*n_minor/n_total:.1f}%)',
        '',
        f'Mean {focal_k} fraction across these MD5s: {mean_focal_frac:.3f}',
        '',
        f'Of MD5s with {focal_k} primary, model predicts {focal_k} in:',
        f'  {n_focal_primary_correct} / {n_focal_primary_with_pred} '
        f'({100*n_focal_primary_correct/max(n_focal_primary_with_pred,1):.1f}%)',
        '',
        'Distribution of #distinct K-types per MD5:',
    ]
    for n_k, count in sorted(distinct_k_dist.items()):
        summary_lines.append(f'  {n_k:>3} K-types: {count}')

    summary = '\n'.join(summary_lines)
    print('\n' + summary)

    out_txt = os.path.join(OUTPUT_DIR, f'focal_{focal_k}_summary.txt')
    with open(out_txt, 'w') as f:
        f.write(summary + '\n')
    print(f'\nSaved: {out_txt}')


if __name__ == '__main__':
    main()
