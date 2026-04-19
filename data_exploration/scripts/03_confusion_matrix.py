"""K-type confusion matrix from test-split predictions.

Outputs:
    data_exploration/output/confusion_matrix_K.csv      (full N×N)
    data_exploration/output/top_confusions_K.csv        (top-30 wrong predictions)
    Same for O.
"""

import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from _common import (
    OUTPUT_DIR, best_experiment_dir, load_training_data,
    load_predictor_for_inference, load_test_embeddings, predict_test_split,
)


def confusion_matrix(test_results):
    """Build {true_class: Counter({pred_class: count})}."""
    matrix = defaultdict(Counter)
    for r in test_results:
        matrix[r['true_class']][r['pred_class']] += 1
    return matrix


def write_full_matrix(matrix, classes, path):
    """Write matrix as CSV with rows=true, cols=predicted."""
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['true \\ pred'] + classes)
        for true_cls in classes:
            row = [true_cls]
            for pred_cls in classes:
                row.append(matrix.get(true_cls, {}).get(pred_cls, 0))
            writer.writerow(row)


def write_top_confusions(matrix, path, top_n=30):
    """Write the top-N (true, predicted, count) where true != predicted."""
    confusions = []
    for true_cls, preds in matrix.items():
        true_total = sum(preds.values())
        for pred_cls, n in preds.items():
            if pred_cls == true_cls:
                continue
            confusions.append({
                'true_class': true_cls,
                'predicted_class': pred_cls,
                'count': n,
                'fraction_of_true': n / true_total if true_total > 0 else 0,
            })
    confusions.sort(key=lambda r: -r['count'])
    confusions = confusions[:top_n]

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['true_class', 'predicted_class', 'count', 'fraction_of_true'])
        writer.writeheader()
        for r in confusions:
            r['fraction_of_true'] = f'{r["fraction_of_true"]:.3f}'
            writer.writerow(r)


def main():
    exp_dir = best_experiment_dir()
    print(f'Using experiment: {os.path.basename(exp_dir)}')

    td, splits_k, splits_o = load_training_data(exp_dir)
    predictor = load_predictor_for_inference(exp_dir)
    emb_dict = load_test_embeddings(exp_dir, td, splits_k, splits_o)

    print('Inferring K head on test split...')
    test_k = predict_test_split(predictor, td, splits_k, emb_dict, head='k')
    print(f'  {len(test_k)} predictions')

    print('Inferring O head on test split...')
    test_o = predict_test_split(predictor, td, splits_o, emb_dict, head='o')
    print(f'  {len(test_o)} predictions')

    # K confusion
    matrix_k = confusion_matrix(test_k)
    write_full_matrix(matrix_k, td.k_classes,
                      os.path.join(OUTPUT_DIR, 'confusion_matrix_K.csv'))
    write_top_confusions(matrix_k,
                          os.path.join(OUTPUT_DIR, 'top_confusions_K.csv'))
    print(f'\nSaved: {OUTPUT_DIR}/confusion_matrix_K.csv ({len(td.k_classes)}x{len(td.k_classes)})')
    print(f'Saved: {OUTPUT_DIR}/top_confusions_K.csv (top 30)')

    # O confusion
    matrix_o = confusion_matrix(test_o)
    write_full_matrix(matrix_o, td.o_classes,
                      os.path.join(OUTPUT_DIR, 'confusion_matrix_O.csv'))
    write_top_confusions(matrix_o,
                          os.path.join(OUTPUT_DIR, 'top_confusions_O.csv'))
    print(f'Saved: {OUTPUT_DIR}/confusion_matrix_O.csv ({len(td.o_classes)}x{len(td.o_classes)})')
    print(f'Saved: {OUTPUT_DIR}/top_confusions_O.csv (top 30)')

    # Print top-10 K confusions
    print('\nTop 10 K confusions (true → predicted, count):')
    confusions = []
    for true_cls, preds in matrix_k.items():
        for pred_cls, n in preds.items():
            if pred_cls != true_cls:
                confusions.append((true_cls, pred_cls, n))
    confusions.sort(key=lambda x: -x[2])
    for t, p, n in confusions[:10]:
        print(f'  {t:<10} -> {p:<10}  {n}')


if __name__ == '__main__':
    main()
