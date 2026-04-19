"""PHL error analysis: Are wrong predictions structured confusions or random?

For each phage in PhageHostLearn, gets the model's host ranking and compares
the predicted K-type (top-ranked host's K) vs the true K-type.

Questions answered:
1. When the model is wrong, does it confuse specific K-type pairs consistently?
2. How often does the model predict the right K-type but wrong specific host?
3. Is the confusion matrix sparse (structured) or dense (random)?

Outputs:
    data_exploration/output/phl_error_analysis.txt
    data_exploration/output/phl_confusion_pairs.csv
    data_exploration/output/phl_per_phage_predictions.csv
"""

import csv
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import REPO_ROOT, OUTPUT_DIR


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', default=None,
                        help='Experiment dir (default: best per_tool experiment)')
    args = parser.parse_args()

    from _common import best_experiment_dir
    if args.experiment:
        exp_dir = os.path.abspath(args.experiment)
    else:
        exp_dir = best_experiment_dir()

    print(f'Experiment: {os.path.basename(exp_dir)}')

    # Load predictor
    from cipher.evaluation.runner import load_predictor, load_validation_data
    import yaml

    predictor = load_predictor(exp_dir)

    with open(os.path.join(exp_dir, 'config.yaml')) as f:
        config = yaml.safe_load(f)
    emb_type = config.get('data', {}).get('embedding_type', 'esm2_650m')

    # Load shared validation data (embeddings + pid_md5) the same way runner does
    data_dir = os.path.join(REPO_ROOT, 'data')
    val_data = load_validation_data(data_dir, emb_type)
    emb_dict = val_data['emb_dict']
    pid_md5 = val_data['pid_md5']
    print(f'  {len(emb_dict)} embeddings, {len(pid_md5)} pid->md5 mappings')

    # Load PHL interaction pairs
    from cipher.data.interactions import (
        load_interaction_pairs, load_phage_protein_mapping,
    )
    phl_dir = os.path.join(REPO_ROOT, 'data', 'validation_data',
                           'HOST_RANGE', 'PhageHostLearn')
    pairs = load_interaction_pairs(phl_dir)
    print(f'Loaded {len(pairs)} PHL pairs')

    # Load phage protein mapping
    pm_path = os.path.join(phl_dir, 'metadata', 'phage_protein_mapping.csv')
    phage_protein_map = load_phage_protein_mapping(pm_path)
    print(f'  {len(phage_protein_map)} phages with protein lists')

    # Build structures from pairs
    interactions = defaultdict(dict)
    serotypes = {}
    for p in pairs:
        interactions[p['phage_id']][p['host_id']] = p['label']
        serotypes[p['host_id']] = {'K': p['host_K'], 'O': p['host_O']}

    all_phages = sorted(interactions.keys())
    print(f'  {len(all_phages)} phages, {len(serotypes)} hosts')

    # For each phage, score all candidate hosts and analyze predictions
    from cipher.evaluation.ranking import rank_hosts

    n_correct_host = 0
    n_correct_k = 0
    n_wrong = 0
    n_skipped = 0

    confusion_pairs = Counter()  # (true_K, predicted_K) -> count
    results = []

    for i, phage in enumerate(all_phages):
        if (i + 1) % 50 == 0:
            print(f'  Scoring phage {i+1}/{len(all_phages)}...')

        pos_hosts = [h for h, label in interactions[phage].items() if label == 1]
        candidates = list(interactions[phage].keys())
        if not pos_hosts:
            n_skipped += 1
            continue

        true_k_types = set(serotypes[h]['K'] for h in pos_hosts if h in serotypes)

        proteins = phage_protein_map.get(phage, set())
        ranked = rank_hosts(predictor, proteins, candidates, serotypes,
                            emb_dict, pid_md5)
        if not ranked:
            n_skipped += 1
            continue

        # Top predicted host
        pred_hid, pred_score = ranked[0]
        pred_k = serotypes[pred_hid]['K']

        # Primary true K
        primary_true_k = Counter(
            serotypes[h]['K'] for h in pos_hosts if h in serotypes
        ).most_common(1)[0][0]

        if pred_hid in set(pos_hosts):
            n_correct_host += 1
            category = 'correct_host'
        elif pred_k in true_k_types:
            n_correct_k += 1
            category = 'correct_k'
        else:
            n_wrong += 1
            confusion_pairs[(primary_true_k, pred_k)] += 1
            category = 'wrong_k'

        # Top-5 unique predicted K-types
        top5_k = []
        seen_k = set()
        for hid, score in ranked:
            k = serotypes[hid]['K']
            if k not in seen_k:
                seen_k.add(k)
                top5_k.append((k, score))
                if len(top5_k) >= 5:
                    break

        results.append({
            'phage_id': phage,
            'true_k_types': ','.join(sorted(true_k_types)),
            'primary_true_k': primary_true_k,
            'pred_k_1': top5_k[0][0] if len(top5_k) > 0 else '',
            'pred_k_2': top5_k[1][0] if len(top5_k) > 1 else '',
            'pred_k_3': top5_k[2][0] if len(top5_k) > 2 else '',
            'n_candidates': len(candidates),
            'n_true_hosts': len(pos_hosts),
            'category': category,
        })

    n_total = n_correct_host + n_correct_k + n_wrong
    print(f'\n{"="*70}')
    print(f'PHL ERROR ANALYSIS — {os.path.basename(exp_dir)}')
    print(f'{"="*70}')

    lines = []
    lines.append(f'Total phages scored: {n_total}  (skipped {n_skipped})')
    lines.append(f'')
    lines.append(f'  Correct host (exact match):      {n_correct_host:>4}  ({100*n_correct_host/n_total:.1f}%)')
    lines.append(f'  Wrong host, correct K-type:      {n_correct_k:>4}  ({100*n_correct_k/n_total:.1f}%)')
    lines.append(f'  Wrong K-type entirely:           {n_wrong:>4}  ({100*n_wrong/n_total:.1f}%)')
    lines.append(f'')

    # Analyze confusion structure
    n_unique_confusions = len(confusion_pairs)

    lines.append(f'Error concentration:')
    lines.append(f'  {n_wrong} wrong-K errors spread across {n_unique_confusions} unique (true->pred) pairs')
    if n_wrong > 0:
        top5_count = sum(c for _, c in confusion_pairs.most_common(5))
        top10_count = sum(c for _, c in confusion_pairs.most_common(10))
        top20_count = sum(c for _, c in confusion_pairs.most_common(20))
        lines.append(f'  Top 5 confusions account for:    {top5_count:>4}  ({100*top5_count/n_wrong:.1f}% of errors)')
        lines.append(f'  Top 10 confusions account for:   {top10_count:>4}  ({100*top10_count/n_wrong:.1f}% of errors)')
        lines.append(f'  Top 20 confusions account for:   {top20_count:>4}  ({100*top20_count/n_wrong:.1f}% of errors)')

    # What K-types does the model predict most often when wrong?
    pred_k_when_wrong = Counter()
    for (true_k, pred_k), count in confusion_pairs.items():
        pred_k_when_wrong[pred_k] += count

    lines.append(f'')
    lines.append(f'Most commonly predicted K-types when wrong:')
    lines.append(f'  {"Pred K":<12} {"Times predicted":>15}  {"% of errors":>10}')
    lines.append(f'  {"-"*12} {"-"*15}  {"-"*10}')
    for pred_k, count in pred_k_when_wrong.most_common(15):
        pct = 100 * count / n_wrong if n_wrong else 0
        lines.append(f'  {pred_k:<12} {count:>15}  {pct:>9.1f}%')

    lines.append(f'')
    lines.append(f'Top 20 confusions (true_K -> predicted_K):')
    lines.append(f'  {"True K":<12} {"Pred K":<12} {"Count":>6}  {"% of errors":>10}')
    lines.append(f'  {"-"*12} {"-"*12} {"-"*6}  {"-"*10}')
    for (true_k, pred_k), count in confusion_pairs.most_common(20):
        pct = 100 * count / n_wrong if n_wrong else 0
        lines.append(f'  {true_k:<12} {pred_k:<12} {count:>6}  {pct:>9.1f}%')

    # Per true-K-type: are errors concentrated or scattered?
    lines.append(f'')
    lines.append(f'Per true-K-type error patterns (is the model confused or random?):')
    lines.append(f'  {"True K":<12} {"Errors":>6}  {"Unique preds":>12}  {"Top pred":>12}  {"Top count":>9}  {"Concentrated?":>13}')
    lines.append(f'  {"-"*12} {"-"*6}  {"-"*12}  {"-"*12}  {"-"*9}  {"-"*13}')

    true_k_errors = defaultdict(Counter)
    for (true_k, pred_k), count in confusion_pairs.items():
        true_k_errors[true_k][pred_k] += count

    for true_k in sorted(true_k_errors, key=lambda k: -sum(true_k_errors[k].values())):
        preds = true_k_errors[true_k]
        total_err = sum(preds.values())
        n_unique = len(preds)
        top_pred, top_count = preds.most_common(1)[0]
        concentration = top_count / total_err if total_err else 0
        label = 'YES' if concentration > 0.5 else ('moderate' if concentration > 0.3 else 'scattered')
        lines.append(f'  {true_k:<12} {total_err:>6}  {n_unique:>12}  {top_pred:>12}  {top_count:>9}  {label:>13}')

    # Randomness check
    lines.append(f'')
    lines.append(f'Randomness check:')
    # Get number of K classes in the model
    sample_pid = list(phage_protein_map.values())[0]
    sample_md5 = [pid_md5[p] for p in sample_pid if p in pid_md5]
    if sample_md5 and sample_md5[0] in emb_dict:
        n_trainable_k = len(predictor.predict_protein(emb_dict[sample_md5[0]])['k_probs'])
        lines.append(f'  Model has {n_trainable_k} K-type classes')
    else:
        n_trainable_k = None
    lines.append(f'  {n_unique_confusions} unique (true->pred) error pairs observed')
    if n_wrong > 0 and n_trainable_k:
        top_pred_k, top_pred_count = pred_k_when_wrong.most_common(1)[0]
        expected_random = n_wrong / n_trainable_k
        lines.append(f'  Most common wrong prediction: {top_pred_k} ({top_pred_count} times, {100*top_pred_count/n_wrong:.1f}%)')
        lines.append(f'  If random, each pred K would appear ~{expected_random:.1f} times')
        lines.append(f'  Actual top is {top_pred_count/max(expected_random,0.1):.1f}x the random baseline')

    summary = '\n'.join(lines)
    print(summary)

    # Save outputs
    txt_path = os.path.join(OUTPUT_DIR, 'phl_error_analysis.txt')
    with open(txt_path, 'w') as f:
        f.write(f'PHL ERROR ANALYSIS — {os.path.basename(exp_dir)}\n')
        f.write('=' * 70 + '\n')
        f.write(summary + '\n')
    print(f'\nSaved: {txt_path}')

    csv_path = os.path.join(OUTPUT_DIR, 'phl_confusion_pairs.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true_K', 'predicted_K', 'count'])
        for (true_k, pred_k), count in confusion_pairs.most_common():
            writer.writerow([true_k, pred_k, count])
    print(f'Saved: {csv_path}')

    results_path = os.path.join(OUTPUT_DIR, 'phl_per_phage_predictions.csv')
    with open(results_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f'Saved: {results_path}')


if __name__ == '__main__':
    main()
