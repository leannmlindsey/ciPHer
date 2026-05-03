"""Training script for MultiTowerAttentionMLP.

Three-tower mid-fusion: per-tower SE-attention + MLP -> 320-d latent each;
concat to 960-d; one K head and one O head share the joint representation.

Training data:
    Per-head positive lists (e.g. v3_uat: HC_K_UAT_multitop +
    HC_O_UAT_multitop). Each row in the training set may be in K-list,
    O-list, both, or neither (we drop "neither"). The shared model trains
    on (K-list union O-list) intersected with proteins that have all
    three embedding files. Per-row K-mask / O-mask gate the BCE losses
    so a protein contributes only to the heads its label is trustworthy
    for. Sum of (per-K-row-mean K-loss) and (per-O-row-mean O-loss),
    equally weighted.

This script's structure mirrors the existing
models/light_attention_binary/train.py per-head data prep helper, but
the model has shared towers so we only train one model rather than two
heads in one job.
"""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from cipher.data.training import TrainingConfig, prepare_training_data
from cipher.data.embeddings import load_embeddings

from model import (  # noqa: E402
    MultiTowerAttentionMLP,
    MultiTowerDataset,
    multi_tower_collate,
    compute_metrics,
    DEFAULT_TOWER_HIDDEN_DIMS,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_single_head_training_config_dict(base_exp_cfg, head_positive_list_path):
    """Strip per-head + tool-filter fields when collapsing to a single
    positive_list call. Same helper as light_attention_binary; copied here
    so this model is self-contained."""
    head_cfg = dict(base_exp_cfg)
    head_cfg['positive_list_path'] = head_positive_list_path
    for conflicting in ('positive_list_k_path', 'positive_list_o_path',
                        'tools', 'exclude_tools', 'protein_set'):
        head_cfg.pop(conflicting, None)
    return head_cfg


def _prep_per_head(head_name, base_exp_cfg, positive_list_path,
                   association_map, glycan_binders, experiment_dir):
    """Run prepare_training_data with one positive list. Saves a
    head-suffixed summary so both per-head datasets are inspectable."""
    head_cfg = make_single_head_training_config_dict(
        base_exp_cfg, positive_list_path)
    cfg = TrainingConfig.from_dict(head_cfg)
    td = prepare_training_data(cfg, association_map, glycan_binders)
    out = os.path.join(experiment_dir, f'training_data_{head_name}')
    os.makedirs(out, exist_ok=True)
    td.save(out)
    return td


def _diagnostic_class_coverage(label_matrix, classes, head_name):
    """Print per-class sample counts; flag classes with < 5 positive samples."""
    counts = (label_matrix > 0).sum(axis=0)
    sparse = [(c, n) for c, n in zip(classes, counts) if n < 5]
    print(f'  [{head_name}] class coverage: {len(classes)} classes total, '
          f'{(counts > 0).sum()} have any positives')
    print(f'  [{head_name}] positive-row coverage: '
          f'{(label_matrix > 0).any(axis=1).sum()} / {label_matrix.shape[0]}')
    if sparse:
        msg = ', '.join(f'{c}({n})' for c, n in sparse[:10])
        more = f' (+{len(sparse) - 10} more)' if len(sparse) > 10 else ''
        print(f'  [{head_name}] LOW-SUPPORT (<5 positives): {msg}{more}')


def _random_split(md5s, train_ratio, val_ratio, seed):
    """Random 70/15/15-style shuffle split. Stratification across two
    heads' label structure isn't well-defined for a shared-tower model;
    random with a fixed seed keeps it reproducible."""
    rng = np.random.RandomState(seed)
    perm = list(md5s)
    rng.shuffle(perm)
    n = len(perm)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        'train': perm[:n_train],
        'val': perm[n_train:n_train + n_val],
        'test': perm[n_train + n_val:],
    }


def _masked_bce(logits, targets, mask):
    """BCE per row, sum over classes, then mean over rows where mask=1.
    Returns 0.0 when mask is all-zero (no contributing rows in batch)."""
    per_elem = F.binary_cross_entropy_with_logits(logits, targets,
                                                   reduction='none')
    per_row = per_elem.sum(dim=1)        # [B]
    n = mask.sum().clamp(min=1.0)
    return (per_row * mask).sum() / n


def evaluate(model, loader, device):
    """Run model on a loader, return per-head metrics over rows where the
    head's mask is 1."""
    model.eval()
    all_K_logits, all_O_logits = [], []
    all_K_labels, all_O_labels = [], []
    all_K_mask, all_O_mask = [], []
    with torch.no_grad():
        for xs, k_labels, o_labels, k_mask, o_mask in loader:
            xs = [x.to(device) for x in xs]
            logK, logO = model(xs)
            all_K_logits.append(logK.cpu()); all_O_logits.append(logO.cpu())
            all_K_labels.append(k_labels);   all_O_labels.append(o_labels)
            all_K_mask.append(k_mask);       all_O_mask.append(o_mask)
    K_logits = torch.cat(all_K_logits); O_logits = torch.cat(all_O_logits)
    K_labels = torch.cat(all_K_labels); O_labels = torch.cat(all_O_labels)
    K_mask = torch.cat(all_K_mask);     O_mask = torch.cat(all_O_mask)
    K_idx = K_mask.bool(); O_idx = O_mask.bool()
    K_metrics = compute_metrics(K_logits[K_idx], K_labels[K_idx]) \
        if K_idx.any() else {}
    O_metrics = compute_metrics(O_logits[O_idx], O_labels[O_idx]) \
        if O_idx.any() else {}
    return K_metrics, O_metrics


class _MaskedDataset(MultiTowerDataset):
    """Same as MultiTowerDataset but also returns per-row K-mask and O-mask."""

    def __init__(self, md5s, emb_dicts, k_labels, o_labels, k_mask, o_mask):
        super().__init__(md5s, emb_dicts, k_labels, o_labels)
        self.k_mask = torch.as_tensor(np.asarray(k_mask), dtype=torch.float32)
        self.o_mask = torch.as_tensor(np.asarray(o_mask), dtype=torch.float32)

    def __getitem__(self, idx):
        xs, kl, ol = super().__getitem__(idx)
        return xs, kl, ol, self.k_mask[idx], self.o_mask[idx]


def _masked_collate(batch):
    xs_per_sample, kls, ols, kms, oms = zip(*batch)
    n_towers = len(xs_per_sample[0])
    xs_stacked = [torch.stack([s[i] for s in xs_per_sample], dim=0)
                  for i in range(n_towers)]
    return (xs_stacked,
            torch.stack(kls, dim=0),
            torch.stack(ols, dim=0),
            torch.stack(kms, dim=0),
            torch.stack(oms, dim=0))


def train(experiment_dir, config):
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    exp_cfg = config.get('experiment', {})
    seed = int(train_cfg.get('seed', 42))

    # Per-head training lists are required.
    pl_k = data_cfg.get('positive_list_k_path') or exp_cfg.get('positive_list_k_path')
    pl_o = data_cfg.get('positive_list_o_path') or exp_cfg.get('positive_list_o_path')
    if not (pl_k and pl_o):
        raise ValueError(
            'multi_tower_attention_mlp requires per-head positive lists. '
            'Set both positive_list_k_path and positive_list_o_path '
            '(e.g. HC_K_UAT_multitop.list and HC_O_UAT_multitop.list).')

    # Three embedding files, in fixed tower order.
    emb_files = [
        data_cfg['embedding_file_a'],   # tower A: ProtT5 XL seg8
        data_cfg['embedding_file_b'],   # tower B: ESM-2 3B mean
        data_cfg['embedding_file_c'],   # tower C: kmer aa20 k4
    ]

    print('=' * 70)
    print('TRAINING MultiTowerAttentionMLP')
    print('=' * 70)
    print(f'  Experiment dir: {experiment_dir}')
    print(f'  Seed: {seed}')
    print(f'  K positive list: {pl_k}')
    print(f'  O positive list: {pl_o}')
    for i, p in enumerate(emb_files):
        print(f'  Tower {chr(ord("A") + i)} embedding: {p}')

    print('\nStep 1: Preparing training data (twice; one per head)...')
    assoc_map = data_cfg.get(
        'association_map',
        'data/training_data/metadata/host_phage_protein_map.tsv')
    glycan = data_cfg.get(
        'glycan_binders',
        'data/training_data/metadata/glycan_binders_custom.tsv')
    td_k = _prep_per_head('k', exp_cfg, pl_k, assoc_map, glycan, experiment_dir)
    td_o = _prep_per_head('o', exp_cfg, pl_o, assoc_map, glycan, experiment_dir)
    print(f'  K dataset: {len(td_k.md5_list)} MD5s, '
          f'{len(td_k.k_classes)} K classes')
    print(f'  O dataset: {len(td_o.md5_list)} MD5s, '
          f'{len(td_o.o_classes)} O classes')

    K_classes = td_k.k_classes
    O_classes = td_o.o_classes
    md5_to_klabel = {m: td_k.k_labels[i] for i, m in enumerate(td_k.md5_list)}
    md5_to_olabel = {m: td_o.o_labels[i] for i, m in enumerate(td_o.md5_list)}

    # Universe = (K-list union O-list)
    universe = set(td_k.md5_list) | set(td_o.md5_list)
    print(f'  Universe (K-list union O-list): {len(universe)} MD5s')

    print('\nStep 2: Loading 3 embedding files and intersecting...')
    emb_dicts = []
    for i, p in enumerate(emb_files):
        d = load_embeddings(p, md5_filter=universe)
        emb_dicts.append(d)
        print(f'  Tower {chr(ord("A") + i)}: loaded {len(d)} embeddings '
              f'(of {len(universe)} in universe)')

    final_md5s = sorted(
        set(emb_dicts[0]) & set(emb_dicts[1]) & set(emb_dicts[2]) & universe
    )
    print(f'  Intersection (universe AND all 3 embeddings): {len(final_md5s)} MD5s')

    if len(final_md5s) < 100:
        raise RuntimeError(
            f'Intersection too small ({len(final_md5s)} MD5s); something is '
            f'misaligned across the 3 embedding files or the per-head lists.')

    # Build label matrices and per-row masks.
    K_labels = np.zeros((len(final_md5s), len(K_classes)), dtype=np.float32)
    O_labels = np.zeros((len(final_md5s), len(O_classes)), dtype=np.float32)
    K_mask = np.zeros(len(final_md5s), dtype=np.float32)
    O_mask = np.zeros(len(final_md5s), dtype=np.float32)
    for i, m in enumerate(final_md5s):
        if m in md5_to_klabel:
            K_labels[i] = md5_to_klabel[m]
            K_mask[i] = 1.0
        if m in md5_to_olabel:
            O_labels[i] = md5_to_olabel[m]
            O_mask[i] = 1.0

    print(f'\nStep 3: Class coverage diagnostic')
    _diagnostic_class_coverage(K_labels[K_mask > 0], K_classes, 'K')
    _diagnostic_class_coverage(O_labels[O_mask > 0], O_classes, 'O')
    print(f'  K-mask=1 rows: {int(K_mask.sum())} / {len(final_md5s)}')
    print(f'  O-mask=1 rows: {int(O_mask.sum())} / {len(final_md5s)}')

    print('\nStep 4: Creating splits...')
    splits = _random_split(
        final_md5s,
        train_ratio=train_cfg.get('train_ratio', 0.7),
        val_ratio=train_cfg.get('val_ratio', 0.15),
        seed=seed,
    )
    md5_to_idx = {m: i for i, m in enumerate(final_md5s)}
    print(f'  Train: {len(splits["train"])}  Val: {len(splits["val"])}  '
          f'Test: {len(splits["test"])}')
    with open(os.path.join(experiment_dir, 'splits.json'), 'w') as f:
        json.dump(splits, f, indent=2)

    def make_ds(split_md5s):
        idxs = [md5_to_idx[m] for m in split_md5s]
        return _MaskedDataset(
            md5s=[final_md5s[i] for i in idxs],
            emb_dicts=emb_dicts,
            k_labels=K_labels[idxs],
            o_labels=O_labels[idxs],
            k_mask=K_mask[idxs],
            o_mask=O_mask[idxs],
        )

    train_ds = make_ds(splits['train'])
    val_ds = make_ds(splits['val'])
    test_ds = make_ds(splits['test'])

    input_dims = train_ds.input_dims()
    print(f'  Tower input dims: {input_dims}')

    # Hyperparameters
    model_cfg = config.get('model', {})
    tower_hidden_dims = tuple(model_cfg.get('tower_hidden_dims',
                                            DEFAULT_TOWER_HIDDEN_DIMS))
    se_dim = int(model_cfg.get('se_dim', 640))
    dropout = float(model_cfg.get('dropout', 0.1))
    lr = float(train_cfg.get('learning_rate', 1e-4))
    weight_decay = float(train_cfg.get('weight_decay', 0.01))
    epochs = int(train_cfg.get('epochs', 200))
    patience = int(train_cfg.get('patience', 30))
    batch_size = int(train_cfg.get('batch_size', 64))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\nStep 5: Building model on {device}')
    model = MultiTowerAttentionMLP(
        input_dims=input_dims,
        num_K=len(K_classes),
        num_O=len(O_classes),
        tower_hidden_dims=tower_hidden_dims,
        se_dim=se_dim,
        dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Trainable params: {n_params:,}')

    pin = torch.cuda.is_available()
    workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=workers,
                              pin_memory=pin, collate_fn=_masked_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=pin,
                            collate_fn=_masked_collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=workers, pin_memory=pin,
                             collate_fn=_masked_collate)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    print('\nStep 6: Training...')
    set_seed(seed)
    history = {'train': [], 'val': [], 'test': []}
    best_val_metric = -float('inf')
    patience_counter = 0
    best_path = os.path.join(experiment_dir, 'best_model.pt')

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kw): return x

    epoch_iter = tqdm(range(epochs), desc='multi_tower', unit='epoch')
    for epoch in epoch_iter:
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for xs, kls, ols, kms, oms in train_loader:
            xs = [x.to(device) for x in xs]
            kls = kls.to(device); ols = ols.to(device)
            kms = kms.to(device); oms = oms.to(device)
            optimizer.zero_grad()
            logK, logO = model(xs)
            loss_K = _masked_bce(logK, kls, kms)
            loss_O = _masked_bce(logO, ols, oms)
            loss = loss_K + loss_O
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        K_val, O_val = evaluate(model, val_loader, device)
        K_test, O_test = evaluate(model, test_loader, device)
        history['train'].append({'loss': avg_loss})
        history['val'].append({'K': K_val, 'O': O_val})
        history['test'].append({'K': K_test, 'O': O_test})

        # Early-stopping metric: average of K f1_macro and O f1_macro
        # (both heads matter; jointly improving them is the goal).
        cur = 0.5 * (K_val.get('f1_macro', 0.0) + O_val.get('f1_macro', 0.0))
        if hasattr(epoch_iter, 'set_postfix'):
            epoch_iter.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'K_f1m': f'{K_val.get("f1_macro", 0):.3f}',
                'O_f1m': f'{O_val.get("f1_macro", 0):.3f}',
                'best': f'{best_val_metric:.3f}' if best_val_metric > -float('inf') else '-',
                'pat': patience_counter,
            })
        if cur > best_val_metric:
            best_val_metric = cur
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch + 1}')
            break

    # Save model + experiment metadata
    head_config = {
        'model': 'multi_tower_attention_mlp',
        'input_dims': input_dims,
        'tower_hidden_dims': list(tower_hidden_dims),
        'se_dim': se_dim,
        'dropout': dropout,
        'num_K': len(K_classes),
        'num_O': len(O_classes),
        'K_classes': K_classes,
        'O_classes': O_classes,
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'patience': patience,
        'batch_size': batch_size,
        'seed': seed,
        'positive_list_k_path': pl_k,
        'positive_list_o_path': pl_o,
        'embedding_file_a': emb_files[0],
        'embedding_file_b': emb_files[1],
        'embedding_file_c': emb_files[2],
        'n_train': len(splits['train']),
        'n_val': len(splits['val']),
        'n_test': len(splits['test']),
        'best_val_metric': float(best_val_metric),
        'best_epoch': len(history['train']) - patience_counter,
    }
    with open(os.path.join(experiment_dir, 'config.json'), 'w') as f:
        json.dump(head_config, f, indent=2)
    with open(os.path.join(experiment_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    try:
        from cipher.provenance import capture_provenance
        prov = capture_provenance()
    except Exception:
        prov = None
    experiment_meta = {
        'model': 'multi_tower_attention_mlp',
        'config': config,
        'n_K_classes': len(K_classes),
        'n_O_classes': len(O_classes),
        'n_train_md5s': len(splits['train']),
        'tower_input_dims': input_dims,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'provenance': prov,
    }
    with open(os.path.join(experiment_dir, 'experiment.json'), 'w') as f:
        json.dump(experiment_meta, f, indent=2)
    print(f'\nTraining complete. Results in {experiment_dir}')
