"""Training loop for the contrastive encoder.

Produces three artefacts in the experiment directory:
    encoder.pt                       — encoder weights (no ArcFace heads)
    contrastive_train_md5.npz        — drop-in replacement for training embeddings
    contrastive_val_md5.npz          — drop-in replacement for validation embeddings

Downstream classifiers (attention_mlp) point their --embedding_file and
--val_embedding_file at these two NPZs and train as usual.
"""

import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add this model dir to sys.path so local imports work regardless of caller.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from model import ContrastiveModel
from sampler import PKClusterSampler

from cipher.data import load_embeddings
from cipher.data.training import TrainingConfig, prepare_training_data


class EmbeddingDataset(Dataset):
    """Thin wrapper around (X, k_label, o_label) tensors."""

    def __init__(self, X, k_labels, o_labels):
        self.X = X
        self.k = k_labels
        self.o = o_labels

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.k[idx], self.o[idx]


def _primary_class_from_multilabel(labels_matrix, ignore_index=-1):
    """For each row, return the argmax column if any column > 0, else ignore_index."""
    has_label = (labels_matrix > 0).any(axis=1)
    primary = np.full(labels_matrix.shape[0], ignore_index, dtype=np.int64)
    if labels_matrix.shape[1] > 0:
        primary[has_label] = labels_matrix[has_label].argmax(axis=1)
    return primary


def _class_cosine_gap(z, labels, rng, within_n=2000, between_n=5000,
                     ignore_index=-1):
    """Quick within-class vs between-class cosine gap on held-out val embeddings.

    Returns (within_mean, between_mean, gap). Used as the early-stopping metric.
    """
    z = z.detach().cpu().numpy()
    labels = np.asarray(labels)
    valid = labels != ignore_index
    z = z[valid]
    labels = labels[valid]
    if len(z) < 4:
        return 0.0, 0.0, 0.0

    # Group indices by class
    by_class = {}
    for i, c in enumerate(labels):
        by_class.setdefault(int(c), []).append(i)
    eligible = [c for c, ix in by_class.items() if len(ix) >= 2]
    if not eligible:
        return 0.0, 0.0, 0.0

    # Within-class pairs
    within = []
    for c in eligible:
        ixs = by_class[c]
        need = max(1, within_n // len(eligible))
        for _ in range(min(need, len(ixs) * (len(ixs) - 1) // 2)):
            a, b = rng.choice(ixs, size=2, replace=False)
            within.append((a, b))
    within = np.array(within) if within else np.zeros((0, 2), dtype=int)

    # Between-class pairs
    all_ix = np.arange(len(z))
    between = []
    while len(between) < between_n:
        a = rng.choice(all_ix, size=between_n * 2)
        b = rng.choice(all_ix, size=between_n * 2)
        for i, j in zip(a, b):
            if i != j and labels[i] != labels[j]:
                between.append((i, j))
                if len(between) >= between_n:
                    break
    between = np.array(between[:between_n])

    w = np.einsum('ij,ij->i', z[within[:, 0]], z[within[:, 1]])
    b = np.einsum('ij,ij->i', z[between[:, 0]], z[between[:, 1]])
    return float(w.mean()), float(b.mean()), float(w.mean() - b.mean())


def _apply_encoder_to_npz(encoder, npz_path, out_path, device, batch_size=512):
    """Load all embeddings from npz_path, run through encoder, save to out_path."""
    encoder.eval()
    data = np.load(npz_path)
    keys = list(data.files)
    if not keys:
        print(f'  (no keys in {npz_path})')
        return

    out = {}
    with torch.no_grad():
        for s in range(0, len(keys), batch_size):
            chunk = keys[s:s + batch_size]
            X = np.stack([data[k] for k in chunk]).astype(np.float32)
            X_t = torch.from_numpy(X).to(device)
            z = encoder(X_t).cpu().numpy()
            for k, v in zip(chunk, z):
                out[k] = v

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f'  wrote {out_path} ({len(out):,} embeddings, dim={z.shape[1]})')


def train(experiment_dir, config):
    """Entry point called by cipher-train."""
    # ---- config ----
    data_cfg = config.get('data', {})
    val_cfg = config.get('validation', {})
    train_cfg = config.get('training', {})
    exp_cfg = config.get('experiment', {})
    model_cfg = config.get('model', {})
    arc_cfg = config.get('arcface', {})
    samp_cfg = config.get('sampler', {})

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = int(train_cfg.get('seed', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    print('=' * 70)
    print('TRAINING ContrastiveEncoder (ArcFace + PK + cluster-stratified)')
    print('=' * 70)
    print(f'  Experiment dir: {experiment_dir}')
    print(f'  Device: {device}')

    # ---- Step 1: prepare_training_data (reuses cipher filtering/sampling) ----
    print('\nStep 1: prepare_training_data ...')
    training_config = TrainingConfig.from_dict(exp_cfg)
    td = prepare_training_data(
        training_config,
        data_cfg.get('association_map',
                     'data/training_data/metadata/host_phage_protein_map.tsv'),
        data_cfg.get('glycan_binders',
                     'data/training_data/metadata/glycan_binders_custom.tsv'),
    )
    td.save(experiment_dir)

    # Primary-K / primary-O per sample (multi-label → single-label for ArcFace).
    primary_k = _primary_class_from_multilabel(td.k_labels)
    primary_o = _primary_class_from_multilabel(td.o_labels)
    n_k_classes = len(td.k_classes)
    n_o_classes = len(td.o_classes)
    print(f'  n samples: {len(td.md5_list):,}')
    print(f'  K-labelled: {(primary_k != -1).sum():,} across {n_k_classes} classes')
    print(f'  O-labelled: {(primary_o != -1).sum():,} across {n_o_classes} classes')

    # ---- Step 2: Load input embeddings ----
    print('\nStep 2: Loading embeddings...')
    emb_file = data_cfg.get('embedding_file',
                            'data/training_data/embeddings/esm2_650m_md5.npz')
    md5_set = set(td.md5_list)
    emb_dict = load_embeddings(emb_file, md5_filter=md5_set)
    # Drop samples we don't have an embedding for
    keep_idx = [i for i, m in enumerate(td.md5_list) if m in emb_dict]
    if len(keep_idx) < len(td.md5_list):
        print(f'  dropped {len(td.md5_list) - len(keep_idx)} samples without embeddings')
    md5_list = [td.md5_list[i] for i in keep_idx]
    primary_k = primary_k[keep_idx]
    primary_o = primary_o[keep_idx]
    X = np.stack([emb_dict[m] for m in md5_list]).astype(np.float32)
    input_dim = X.shape[1]
    print(f'  X: {X.shape}')

    # ---- Step 3: Build cluster ids per sample (for sampler) ----
    cluster_ids = None
    if training_config.cluster_file_path:
        # Reuse the same cluster file loader that _downsample uses.
        from cipher.data.training import _build_md5_cluster_map
        # We need `rows` to map md5 -> cluster_id. Re-load association map.
        from cipher.data.interactions import load_training_map
        rows = load_training_map(data_cfg.get(
            'association_map',
            'data/training_data/metadata/host_phage_protein_map.tsv'))
        md5_to_cluster = _build_md5_cluster_map(
            training_config.cluster_file_path,
            training_config.cluster_threshold, rows,
            log=lambda s: print('  ' + s))
        cluster_ids = [md5_to_cluster.get(m) for m in md5_list]
        # md5s without cluster get None → singleton clusters in the sampler
        cluster_ids = [c if c is not None else f'_solo_{m}'
                       for c, m in zip(cluster_ids, md5_list)]
        print(f'  cluster-stratified sampling enabled '
              f'(threshold=cl{training_config.cluster_threshold})')

    # ---- Step 4: train/val split (internal, for early stopping) ----
    rng = np.random.default_rng(seed)
    n = len(md5_list)
    perm = rng.permutation(n)
    n_train = int(n * float(train_cfg.get('train_ratio', 0.85)))
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]
    print(f'\nStep 3: internal split — train: {len(train_idx)}, val: {len(val_idx)}')

    X_train = torch.from_numpy(X[train_idx])
    X_val = torch.from_numpy(X[val_idx])
    k_train = torch.from_numpy(primary_k[train_idx]).long()
    k_val = torch.from_numpy(primary_k[val_idx]).long()
    o_train = torch.from_numpy(primary_o[train_idx]).long()
    o_val = torch.from_numpy(primary_o[val_idx]).long()

    # ---- Step 5: sampler ----
    # Sampler operates on train-split indices. Label per sample = primary K.
    # We drop samples without a K label from the sampler's pool (they can still
    # contribute via the O head on batches that happen to include them — but
    # since we sample by K, they won't).
    train_primary_k = primary_k[train_idx]
    train_clusters = [cluster_ids[i] for i in train_idx] if cluster_ids else None

    P = int(samp_cfg.get('P', 32))
    K = int(samp_cfg.get('K', 8))
    # Only K-labelled samples go to the sampler; others need a per-sample
    # unique pseudo-class so the usable-class filter (min K samples per class)
    # excludes them cleanly instead of lumping them into one large pool.
    sampler_classes = [
        int(c) if c != -1 else f'_unlabeled_{i}'
        for i, c in enumerate(train_primary_k)
    ]
    sampler = PKClusterSampler(
        class_labels=sampler_classes,
        cluster_ids=train_clusters,
        P=P, K=K,
        num_batches_per_epoch=samp_cfg.get('num_batches_per_epoch'),
        seed=seed,
    )
    print(f'  sampler: P={P}, K={K}, batches/epoch={len(sampler)}')

    train_ds = EmbeddingDataset(X_train, k_train, o_train)
    train_loader = DataLoader(train_ds, batch_sampler=sampler, num_workers=0)

    # ---- Step 6: build model ----
    model = ContrastiveModel(
        input_dim=input_dim,
        n_k_classes=n_k_classes, n_o_classes=n_o_classes,
        hidden_dims=tuple(model_cfg.get('hidden_dims', [1280, 1024, 1024])),
        output_dim=model_cfg.get('output_dim'),
        dropout=float(model_cfg.get('dropout', 0.1)),
        margin=float(arc_cfg.get('margin', 0.5)),
        scale=float(arc_cfg.get('scale', 30.0)),
    ).to(device)

    lambda_k = float(train_cfg.get('lambda_k', 1.0))
    lambda_o = float(train_cfg.get('lambda_o', 1.0))
    # `heads` is orthogonal to lambda_{k,o} — it's a coarser kill-switch
    # exposed via CLI. When set, it overrides lambda to 0 for the dropped
    # head regardless of whether the user also specified a lambda.
    heads = str(train_cfg.get('heads', 'both')).lower()
    if heads not in ('both', 'k', 'o'):
        raise ValueError(f"training.heads must be one of {{'both','k','o'}}, got {heads!r}")
    if heads == 'k':
        lambda_o = 0.0
        print('  heads=k: forcing lambda_o=0')
    elif heads == 'o':
        lambda_k = 0.0
        print('  heads=o: forcing lambda_k=0')
    lr = float(train_cfg.get('learning_rate', 1e-4))
    wd = float(train_cfg.get('weight_decay', 1e-4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # ---- Step 7: training loop ----
    epochs = int(train_cfg.get('epochs', 100))
    patience = int(train_cfg.get('patience', 20))
    hard_neg_on = bool(samp_cfg.get('hard_negative_mining', False))
    hard_neg_start = int(samp_cfg.get('hard_negative_start_epoch', 10))
    hard_neg_every = max(1, int(samp_cfg.get('hard_negative_update_every', 5)))
    hard_neg_temp = float(samp_cfg.get('hard_negative_temperature', 1.0))
    if hard_neg_on:
        print(f'  hard-negative mining: enabled  '
              f'(start_epoch={hard_neg_start}, update_every={hard_neg_every}, '
              f'temperature={hard_neg_temp})')
    history = []
    best_gap = -1.0
    best_epoch = 0
    best_state = None
    since_best = 0

    print(f'\nStep 4: training — epochs={epochs}, patience={patience}, lr={lr}')
    for epoch in range(epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = total_k_loss = total_o_loss = 0.0
        n_batches = 0
        for x, kl, ol in train_loader:
            x = x.to(device, non_blocking=True)
            kl = kl.to(device, non_blocking=True)
            ol = ol.to(device, non_blocking=True)
            out = model(x, kl, ol)
            loss = lambda_k * out['k_loss'] + lambda_o * out['o_loss']
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            total_k_loss += float(out['k_loss'].item())
            total_o_loss += float(out['o_loss'].item())
            n_batches += 1
        mean_loss = total_loss / max(n_batches, 1)

        # Validation: compute within/between cosine gap on K labels.
        model.eval()
        with torch.no_grad():
            z_val = model.encoder(X_val.to(device))
        w_mean, b_mean, gap = _class_cosine_gap(z_val, k_val.numpy(), rng)

        history.append({
            'epoch': epoch,
            'loss': mean_loss,
            'k_loss': total_k_loss / max(n_batches, 1),
            'o_loss': total_o_loss / max(n_batches, 1),
            'val_within': w_mean,
            'val_between': b_mean,
            'val_gap': gap,
        })
        print(f'  epoch {epoch:3d}  loss={mean_loss:.4f} '
              f'k={total_k_loss/max(n_batches,1):.4f} '
              f'o={total_o_loss/max(n_batches,1):.4f}  '
              f'val within={w_mean:.4f} between={b_mean:.4f} gap={gap:+.4f}')

        if gap > best_gap:
            best_gap = gap
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            since_best = 0
        else:
            since_best += 1
            if since_best >= patience:
                print(f'  early stopping at epoch {epoch} (best gap={best_gap:+.4f} @ epoch {best_epoch})')
                break

        # Hard-negative mining: every N epochs after warm-up, re-weight the
        # PK sampler so P classes are drawn proportional to their
        # current-encoder confusability (max cosine to another prototype).
        if (hard_neg_on and epoch >= hard_neg_start
                and (epoch - hard_neg_start) % hard_neg_every == 0):
            with torch.no_grad():
                z_train_np = model.encoder(X_train.to(device)).cpu().numpy()
            labels_np = k_train.numpy()
            by_class = defaultdict(list)
            for i, c in enumerate(labels_np):
                if c >= 0:
                    by_class[int(c)].append(i)
            if len(by_class) >= 2:
                z_norm = z_train_np / (np.linalg.norm(z_train_np, axis=1, keepdims=True) + 1e-8)
                classes = list(by_class.keys())
                protos = np.stack([z_norm[ix].mean(axis=0) for ix in (by_class[c] for c in classes)])
                protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
                sim = protos @ protos.T
                np.fill_diagonal(sim, -np.inf)
                hardness = sim.max(axis=1)
                weights = {int(c): float(h) for c, h in zip(classes, hardness)}
                sampler.set_class_weights(weights, temperature=hard_neg_temp)
                n_over = int((hardness > np.median(hardness)).sum())
                print(f'    [hard-neg] re-weighted {len(weights)} classes '
                      f'(hardness median={np.median(hardness):+.3f}, '
                      f'top-half count={n_over})')

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ---- Step 8: save encoder + generate drop-in NPZs ----
    print('\nStep 5: saving artifacts...')
    torch.save({
        'encoder_state_dict': model.encoder.state_dict(),
        'input_dim': input_dim,
        'output_dim': model.encoder.output_dim,
        'hidden_dims': list(model_cfg.get('hidden_dims', [1280, 1024, 1024])),
        'dropout': float(model_cfg.get('dropout', 0.1)),
    }, os.path.join(experiment_dir, 'encoder.pt'))

    train_npz_out = os.path.join(experiment_dir, 'contrastive_train_md5.npz')
    val_npz_out = os.path.join(experiment_dir, 'contrastive_val_md5.npz')
    print('  applying encoder to training NPZ...')
    _apply_encoder_to_npz(model.encoder, emb_file, train_npz_out, device)
    val_file = val_cfg.get('val_embedding_file')
    if val_file and os.path.exists(val_file):
        print('  applying encoder to validation NPZ...')
        _apply_encoder_to_npz(model.encoder, val_file, val_npz_out, device)
    else:
        print(f'  (validation NPZ {val_file!r} not present — skipping)')

    # ---- Step 9: experiment.json with provenance ----
    from cipher.provenance import capture_provenance
    meta = {
        'model': 'contrastive_encoder',
        'config': config,
        'n_md5s': len(md5_list),
        'n_k_classes': n_k_classes,
        'n_o_classes': n_o_classes,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'provenance': capture_provenance(),
        'best_epoch': best_epoch,
        'best_val_gap': best_gap,
        'history': history,
    }
    with open(os.path.join(experiment_dir, 'experiment.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('\nDone. Downstream usage:')
    print(f'  cipher-train --model attention_mlp \\\n'
          f'      --embedding_file {train_npz_out} \\\n'
          f'      --val_embedding_file {val_npz_out} \\\n'
          f'      [ ... same filter/sampling flags as you used here ... ]')
