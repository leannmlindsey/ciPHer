"""Create stratified train/val/test splits."""

from collections import defaultdict

import numpy as np


# Reused by canonical_split: K-types in this set are treated as "no K signal"
# and the protein is re-clustered by primary O-type instead.
NULL_LABELS = {'null', 'N/A', 'Unknown', '', 'NA', 'n/a',
               'capsule null', 'Capsule null', 'unknown'}


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


def create_canonical_split(md5_list, k_labels, o_labels, k_classes, o_classes,
                            train_ratio=0.7, val_ratio=0.15, seed=42):
    """Create ONE shared train/val/test split usable by both K and O heads.

    Mirrors the old `klebsiella` repo's `create_canonical_split.py`. Each
    MD5 is assigned to a single cluster — primary K-type, falling back to
    `O_fallback:<O>` when K is null/N/A. The canonical per-cluster split
    rules (1-2 → all train, 3 → 2/0/1, 4 → 2/1/1, 5+ → 60/20/20 with min
    3 train and 1 val) are then applied per cluster.

    Why a shared split instead of per-head independent splits:
    With single-label training, K-null + O-valid proteins flow to BOTH
    heads (as a "null" target on K and as a real target on O). Giving them
    the same train/val/test bucket on both heads keeps the early-stopping
    held-out distribution synchronized between heads, reproducing the
    behavior of the old pipeline.

    Args:
        md5_list: list of N protein MD5s.
        k_labels: (N, n_k) one-hot or multi-hot K labels.
        o_labels: (N, n_o) one-hot or multi-hot O labels.
        k_classes: list of K class names (length n_k).
        o_classes: list of O class names (length n_o).
        train_ratio, val_ratio: fractions for ≥10-sample clusters.
        seed: RNG seed.

    Returns:
        dict {'train': [...], 'val': [...], 'test': [...]} — disjoint
        lists of MD5s. Proteins with no K and no O label are excluded.
    """
    rng = np.random.default_rng(seed)

    # Cluster: primary K → cluster name, with O-fallback for null K.
    cluster_to_md5s = defaultdict(list)
    for i, md5 in enumerate(md5_list):
        # Primary K is the argmax of k_labels[i] iff that entry is nonzero.
        if k_labels.shape[1] > 0:
            kmax = int(np.argmax(k_labels[i]))
            primary_k = k_classes[kmax] if k_labels[i, kmax] > 0 else None
        else:
            primary_k = None
        if primary_k is not None and primary_k not in NULL_LABELS:
            cluster_to_md5s[primary_k].append(md5)
            continue
        # K is null/missing — fall back to primary O.
        if o_labels.shape[1] > 0:
            omax = int(np.argmax(o_labels[i]))
            primary_o = o_classes[omax] if o_labels[i, omax] > 0 else None
        else:
            primary_o = None
        if primary_o is not None and primary_o not in NULL_LABELS:
            cluster_to_md5s[f'O_fallback:{primary_o}'].append(md5)
        # else: drop — no K and no O signal to cluster on.

    train, val, test = [], [], []
    for cluster, md5s in cluster_to_md5s.items():
        n = len(md5s)
        shuffled = rng.permutation(md5s).tolist()
        if n <= 2:
            train.extend(shuffled)
        elif n == 3:
            train.extend(shuffled[:2])
            test.extend(shuffled[2:])
        elif n == 4:
            train.extend(shuffled[:2])
            val.extend(shuffled[2:3])
            test.extend(shuffled[3:])
        elif n < 10:
            n_train = max(3, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio))
            train.extend(shuffled[:n_train])
            val.extend(shuffled[n_train:n_train + n_val])
            test.extend(shuffled[n_train + n_val:])
        else:
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train.extend(shuffled[:n_train])
            val.extend(shuffled[n_train:n_train + n_val])
            test.extend(shuffled[n_train + n_val:])

    return {'train': train, 'val': val, 'test': test}
