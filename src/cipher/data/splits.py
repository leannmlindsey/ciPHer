"""Create stratified train/val/test splits."""

from collections import defaultdict

import numpy as np


def create_stratified_split(ids, labels, train_ratio=0.7, val_ratio=0.15,
                             seed=42):
    """Create stratified train/val/test split based on class labels.

    Ensures each class has representation in train, and where possible
    in val and test. Small classes (<=2 samples) go entirely to train.

    Args:
        ids: list of sample IDs
        labels: list of class labels (same length as ids)
        train_ratio: fraction for training
        val_ratio: fraction for validation (test gets the rest)
        seed: random seed

    Returns:
        dict: {'train': [...], 'val': [...], 'test': [...]}
    """
    rng = np.random.default_rng(seed)

    label_to_ids = defaultdict(list)
    for i, label in enumerate(labels):
        label_to_ids[label].append(ids[i])

    train_ids, val_ids, test_ids = [], [], []

    for label in sorted(label_to_ids.keys()):
        members = label_to_ids[label]
        shuffled = rng.permutation(members).tolist()
        n = len(shuffled)

        if n <= 2:
            train_ids.extend(shuffled)
        elif n == 3:
            train_ids.extend(shuffled[:2])
            test_ids.extend(shuffled[2:])
        elif n == 4:
            train_ids.extend(shuffled[:2])
            val_ids.extend(shuffled[2:3])
            test_ids.extend(shuffled[3:])
        else:
            n_train = max(3, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            train_ids.extend(shuffled[:n_train])
            val_ids.extend(shuffled[n_train:n_train + n_val])
            test_ids.extend(shuffled[n_train + n_val:])

    return {'train': train_ids, 'val': val_ids, 'test': test_ids}
