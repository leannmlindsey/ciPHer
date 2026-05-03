"""Training script for LightAttentionBinary serotype prediction model.

Trains one K head and one O head in the same experiment, mirroring the
attention_mlp layout: `model_k/` and `model_o/` subdirectories with
`best_model.pt`, `config.json`, and `training_history.json`.

Per-head pipeline:
    - Independent stratified split on MD5s with K (or O) labels.
    - PerResidueDataset over variable-length ESM embeddings.
    - WeightedRandomSampler on primary class (colleague's approach).
    - BCEWithLogits loss (sum-over-classes, mean-over-batch).
    - AdamW, cosine LR schedule, early stopping on val f1_macro.
"""

import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from cipher.data.training import TrainingConfig, prepare_training_data
from cipher.data.splits import create_stratified_split
from cipher.data.embeddings import load_embeddings

from model import (
    LightAttentionBinary,
    PerResidueDataset,
    per_residue_collate_fn,
    bce_loss,
    compute_metrics,
)


_SUPPORTED_STRATEGIES = {
    'multi_label', 'multi_label_threshold', 'weighted_multi_label',
}


def make_warmup_cosine_schedule(warmup_epochs, total_epochs, min_lr_ratio=2e-3):
    """Linear warmup for `warmup_epochs` then cosine decay over the rest.

    Returns a callable `lambda_fn(epoch) -> lr_factor` suitable for
    torch.optim.lr_scheduler.LambdaLR. `lr_factor` is in
    [min_lr_ratio, 1.0]; the optimizer's effective LR is base_lr * factor.

    With warmup_epochs=0 this reduces to a pure cosine decay matching the
    legacy CosineAnnealingLR(eta_min=base_lr*min_lr_ratio) behaviour.
    Standard recipe for training transformers from scratch -- avoids the
    early-step gradient explosion that locks the model into bad basins.

    Args:
        warmup_epochs: number of epochs to ramp lr linearly 0 -> 1.
        total_epochs: total number of epochs (warmup + decay).
        min_lr_ratio: floor for the lr factor at the end of cosine decay
            (e.g. 2e-3 with base_lr=5e-4 -> floor lr ~ 1e-6).
    """
    def lambda_fn(epoch):
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        decay_epochs = max(1, total_epochs - warmup_epochs)
        progress = min(1.0, (epoch - warmup_epochs) / decay_epochs)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cos_factor
    return lambda_fn


def load_embeddings_or_bins(emb_path, md5_filter=None, log=print):
    """Load embeddings from an NPZ file OR a directory of split-bin NPZs.

    The merge step that combines per-length-bin extraction output into a
    single NPZ has been fragile under disk-quota pressure (see 2026-04-25
    notebook entry: a 199 GiB merged file truncated mid-write twice).
    Loading directly from the bin directory skips the merge entirely,
    saving ~199 GiB of disk per pLM and avoiding the OOM/quota failure mode.

    Behaviour:
        - If `emb_path` is a directory, all `*.npz` files inside are read in
          sorted order and unified into a single dict. Duplicate MD5 keys
          across bins are resolved as "first wins" (this should not happen
          with a clean extraction, but might after a partial resume).
        - If `emb_path` is a file, falls back to `cipher.data.embeddings.
          load_embeddings` (unchanged single-file behaviour).

    Args:
        emb_path: Path to a .npz file, OR a directory containing .npz files.
        md5_filter: Optional iterable of MD5s to keep. Reduces memory
            substantially (~20 GiB instead of ~199 GiB for ESM-2 650M
            per-residue with the highconf filter).
        log: callable for status output (default `print`).

    Returns:
        dict {md5_hash: numpy_array}.
    """
    if not os.path.isdir(emb_path):
        return load_embeddings(emb_path, md5_filter=md5_filter)

    bin_paths = sorted(glob.glob(os.path.join(emb_path, '*.npz')))
    if not bin_paths:
        raise FileNotFoundError(f'No .npz files in directory: {emb_path}')

    md5_filter = set(md5_filter) if md5_filter is not None else None
    out = {}
    for p in bin_paths:
        data = np.load(p)
        n_kept = 0
        n_dup = 0
        for k in data.files:
            if md5_filter is not None and k not in md5_filter:
                continue
            if k in out:
                n_dup += 1
                continue
            out[k] = data[k]
            n_kept += 1
        suffix = f' ({n_dup} duplicates skipped)' if n_dup else ''
        log(f'  bin {os.path.basename(p)}: {n_kept} keys loaded'
            f' (of {len(data.files)} in bin){suffix}')
    return out


def check_embedding_coverage(split_md5s, emb_dict, min_coverage,
                             split_name, emb_file):
    """Raise ValueError if fewer than `min_coverage` of split_md5s have
    embeddings in emb_dict. Returns the list of MD5s that DO have
    embeddings. Silently returns an empty list if split_md5s is empty.

    This guard catches the 14%-coverage failure mode: train.py previously
    would silently drop ~85% of the planned training set when the embedding
    file was partial (see 2026-04-21 notebook entry).
    """
    n_expected = len(split_md5s)
    valid = [m for m in split_md5s if m in emb_dict]
    n_valid = len(valid)
    coverage = n_valid / n_expected if n_expected > 0 else 0.0
    print(f'  {split_name}: {n_valid}/{n_expected} MD5s have embeddings '
          f'({coverage:.1%})')
    if n_expected > 0 and coverage < min_coverage:
        raise ValueError(
            f'Embedding coverage too low for split {split_name!r}: '
            f'{n_valid}/{n_expected} = {coverage:.1%} < '
            f'{min_coverage:.0%}. The embedding file at {emb_file} is '
            f'missing most of the filtered training set. Re-extract '
            f'embeddings for the full candidates set, or lower '
            f'training.min_embedding_coverage in the config to '
            f'acknowledge the gap.'
        )
    return valid


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, loader, device):
    """Run the model over a loader and compute multi-label metrics."""
    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_emb, batch_masks, batch_labels in loader:
            batch_emb = batch_emb.to(device)
            batch_masks = batch_masks.to(device)
            logits = model(batch_emb, batch_masks)
            all_logits.append(logits.cpu())
            all_labels.append(batch_labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return compute_metrics(all_logits, all_labels)


def _weighted_sampler(y_train):
    """Per-colleague: WeightedRandomSampler on primary class frequency.

    For multi-hot targets the "primary class" is argmax of each sample's
    label vector. Inverse-frequency weights balance rare primary classes at
    sampling time rather than through downsampling.
    """
    binary = (y_train > 0).astype(np.int64)
    primary = binary.argmax(axis=1)
    counts = np.bincount(primary, minlength=y_train.shape[1])
    counts = np.maximum(counts, 1)
    class_weight = 1.0 / counts
    sample_weights = class_weight[primary]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_head(head_name, X_train, y_train, X_val, y_val, X_test, y_test,
               classes, config, output_dir):
    """Train one head (K or O) and save best checkpoint + history."""
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(iterable, **kwargs):
            return iterable

    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    strategy = config.get('experiment', {}).get('label_strategy', 'multi_label_threshold')

    if strategy not in _SUPPORTED_STRATEGIES:
        raise ValueError(
            f'light_attention_binary only supports multi-label strategies '
            f'(BCE). Got label_strategy={strategy!r}. Use one of '
            f'{sorted(_SUPPORTED_STRATEGIES)}.'
        )

    pooler_type = model_cfg.get('pooler_type', 'conv_attn')
    pooler_cnn_width = int(model_cfg.get('pooler_cnn_width', 9))
    transformer_num_heads = int(model_cfg.get('transformer_num_heads', 8))
    transformer_num_layers = int(model_cfg.get('transformer_num_layers', 2))
    transformer_max_len = int(model_cfg.get('transformer_max_len', 2048))
    c_terminal_crop = model_cfg.get('c_terminal_crop')
    if c_terminal_crop is not None:
        c_terminal_crop = int(c_terminal_crop)
    dropout = float(model_cfg.get('dropout', 0.1))
    lr = float(train_cfg.get('learning_rate', 5e-4))
    weight_decay = float(train_cfg.get('weight_decay', 0.01))
    epochs = int(train_cfg.get('epochs', 200))
    patience = int(train_cfg.get('patience', 30))
    batch_size = int(train_cfg.get('batch_size', 32))
    grad_clip = float(train_cfg.get('grad_clip', 0.0))
    warmup_epochs = int(train_cfg.get('warmup_epochs', 0))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Training {head_name.upper()} head ({len(classes)} classes)')
    print(f'  Device: {device}')
    print(f'  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

    # Derive embed_dim from the data; `model.embed_dim` in config is only a
    # sanity check. This lets the same code handle ESM-2 650M (D=1280) and
    # ProtT5-XL (D=1024) without a config change.
    embed_dim = X_train[0].shape[1]
    embed_dim_cfg = model_cfg.get('embed_dim')
    if embed_dim_cfg is not None and int(embed_dim_cfg) != embed_dim:
        raise ValueError(
            f'config model.embed_dim={embed_dim_cfg} does not match actual '
            f'embedding dim {embed_dim} from the embedding file.'
        )
    print(f'  embed_dim (auto-detected): {embed_dim}')

    model = LightAttentionBinary(
        embed_dim=embed_dim,
        num_classes=len(classes),
        pooler_type=pooler_type,
        pooler_cnn_width=pooler_cnn_width,
        transformer_num_heads=transformer_num_heads,
        transformer_num_layers=transformer_num_layers,
        transformer_max_len=transformer_max_len,
        dropout=dropout,
    ).to(device)
    print(f'  pooler_type: {pooler_type}')
    if c_terminal_crop:
        print(f'  c_terminal_crop: {c_terminal_crop} residues')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {trainable_params:,} trainable / {total_params:,} total')

    train_dataset = PerResidueDataset(X_train, y_train, c_terminal_crop=c_terminal_crop)
    val_dataset = PerResidueDataset(X_val, y_val, c_terminal_crop=c_terminal_crop)
    test_dataset = PerResidueDataset(X_test, y_test, c_terminal_crop=c_terminal_crop)

    use_pin = torch.cuda.is_available()
    n_workers = 4 if torch.cuda.is_available() else 0

    train_sampler = _weighted_sampler(y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        drop_last=True, num_workers=n_workers, pin_memory=use_pin,
        collate_fn=per_residue_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=n_workers, pin_memory=use_pin,
        collate_fn=per_residue_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=n_workers, pin_memory=use_pin,
        collate_fn=per_residue_collate_fn,
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        eps=1e-6, betas=(0.9, 0.999),
    )
    if warmup_epochs > 0:
        # Linear warmup -> cosine decay. Min-lr factor matches the prior
        # CosineAnnealingLR(eta_min=1e-6) at the default lr=5e-4.
        min_lr_ratio = max(2e-3, 1e-6 / max(lr, 1e-12))
        scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            make_warmup_cosine_schedule(warmup_epochs, epochs, min_lr_ratio),
        )
        print(f'  lr schedule: warmup={warmup_epochs} epochs '
              f'-> cosine decay (min_lr_ratio={min_lr_ratio:.2e})')
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-6)

    os.makedirs(output_dir, exist_ok=True)
    best_val_metric = -float('inf')
    patience_counter = 0
    history = {'train': [], 'val': [], 'test': []}

    epoch_iter = tqdm(range(epochs), desc=f'  {head_name.upper()} head', unit='epoch')
    for epoch in epoch_iter:
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch_emb, batch_masks, batch_labels in train_loader:
            batch_emb = batch_emb.to(device)
            batch_masks = batch_masks.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_emb, batch_masks)
            loss = bce_loss(logits, batch_labels)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = train_loss / max(n_batches, 1)

        val_metrics = evaluate_model(model, val_loader, device)
        test_metrics = evaluate_model(model, test_loader, device)

        history['train'].append({'loss': avg_loss})
        history['val'].append(val_metrics)
        history['test'].append(test_metrics)

        current_metric = val_metrics['f1_macro']

        if hasattr(epoch_iter, 'set_postfix'):
            epoch_iter.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'F1_macro': f'{current_metric:.4f}',
                'F1_micro': f'{val_metrics["f1_micro"]:.4f}',
                'best': f'{best_val_metric:.4f}' if best_val_metric > -float('inf') else '-',
                'pat': patience_counter,
            })

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'  Early stopping at epoch {epoch + 1}')
            break

    head_config = {
        'head': head_name,
        'strategy': strategy,
        'embed_dim': embed_dim,
        'num_classes': len(classes),
        'classes': classes,
        'pooler_type': pooler_type,
        'pooler_cnn_width': pooler_cnn_width,
        'transformer_num_heads': transformer_num_heads,
        'transformer_num_layers': transformer_num_layers,
        'transformer_max_len': transformer_max_len,
        'c_terminal_crop': c_terminal_crop,
        'dropout': dropout,
        'lr': lr,
        'weight_decay': weight_decay,
        'epochs': epochs,
        'patience': patience,
        'warmup_epochs': warmup_epochs,
        'batch_size': batch_size,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'best_epoch': epoch + 1 - patience_counter,
        'best_val_metric': float(best_val_metric),
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(head_config, f, indent=2)
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    best_epoch = epoch + 1 - patience_counter
    best_val = history['val'][best_epoch - 1]
    best_test = history['test'][best_epoch - 1]
    print(f'  Best epoch: {best_epoch}')
    print(f'  Val  F1_macro: {best_val.get("f1_macro", 0):.4f}  '
          f'F1_micro: {best_val.get("f1_micro", 0):.4f}  '
          f'precision_macro: {best_val.get("precision_macro", 0):.4f}  '
          f'recall_macro: {best_val.get("recall_macro", 0):.4f}  '
          f'top1: {best_val.get("top1_acc", 0):.4f}')
    print(f'  Test F1_macro: {best_test.get("f1_macro", 0):.4f}  '
          f'F1_micro: {best_test.get("f1_micro", 0):.4f}  '
          f'precision_macro: {best_test.get("precision_macro", 0):.4f}  '
          f'recall_macro: {best_test.get("recall_macro", 0):.4f}  '
          f'top1: {best_test.get("top1_acc", 0):.4f}')

    return history


def _md5s_with_labels(labels, md5_list):
    """Subset of md5_list whose label rows have at least one nonzero entry."""
    keep = (labels > 0).any(axis=1)
    return [m for m, k in zip(md5_list, keep) if k]


def make_single_head_training_config_dict(base_exp_cfg, head_positive_list_path):
    """Collapse a per-head training config (with positive_list_k/o_path)
    into a single-list config that TrainingConfig.from_dict will accept.

    `TrainingConfig.__post_init__` enforces a mutex: `positive_list_path`
    cannot coexist with `positive_list_k_path`, `positive_list_o_path`,
    `tools`, `exclude_tools`, or `protein_set`. When we run
    prepare_training_data once per head, each call uses a single
    positive_list_path (the per-head list); every conflicting field
    inherited from the experiment-level config has to be stripped first
    or from_dict() raises ValueError. (Hit this 2026-05-03, job 2234134.)

    Returns a NEW dict; does not mutate the input.
    """
    head_cfg = dict(base_exp_cfg)
    if head_positive_list_path:
        head_cfg['positive_list_path'] = head_positive_list_path
        for conflicting in ('positive_list_k_path', 'positive_list_o_path',
                            'tools', 'exclude_tools', 'protein_set'):
            head_cfg.pop(conflicting, None)
    return head_cfg


def _prep_data_for_head(head_name, base_exp_cfg, head_positive_list_path,
                        association_map, glycan_binders, experiment_dir):
    """Run prepare_training_data with a head-specific positive_list override.

    Returns the resulting TrainingData. Saves a head-suffixed copy of the
    summary so both per-head datasets are inspectable in the experiment dir.
    """
    head_cfg = make_single_head_training_config_dict(
        base_exp_cfg, head_positive_list_path)
    training_config = TrainingConfig.from_dict(head_cfg)
    td = prepare_training_data(training_config, association_map, glycan_binders)
    head_dir = os.path.join(experiment_dir, f'training_data_{head_name}')
    os.makedirs(head_dir, exist_ok=True)
    td.save(head_dir)
    return td


def train(experiment_dir, config):
    """Train both K and O heads for the LightAttentionBinary model.

    Two filter modes:
      * Single-list (legacy): one `positive_list` (or tool filter) drives
        the training pool for both heads. K and O training sets are the
        intersection of the same pool with proteins having a K or O label.
      * Per-head: `positive_list_k` and `positive_list_o` are set; each
        head trains on a separately-filtered pool. Required to use the
        v2/v3/v4 highconf-style per-head lists where K and O coverage
        differs deliberately (e.g. HC_K_UAT_multitop + HC_O_UAT_multitop).
    """
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    exp_cfg = config.get('experiment', {})
    seed = train_cfg.get('seed', 42)

    pl_k = data_cfg.get('positive_list_k_path') or exp_cfg.get('positive_list_k_path')
    pl_o = data_cfg.get('positive_list_o_path') or exp_cfg.get('positive_list_o_path')
    per_head_mode = bool(pl_k and pl_o)

    print('=' * 70)
    print('TRAINING LightAttentionBinary')
    print('=' * 70)
    print(f'  Experiment dir: {experiment_dir}')
    print(f'  Seed: {seed}')
    print(f'  Filter mode: {"per-head" if per_head_mode else "single-list"}')
    if per_head_mode:
        print(f'    K positive list: {pl_k}')
        print(f'    O positive list: {pl_o}')

    print('\nStep 1: Preparing training data...')
    assoc_map = data_cfg.get(
        'association_map',
        'data/training_data/metadata/host_phage_protein_map.tsv')
    glycan = data_cfg.get(
        'glycan_binders',
        'data/training_data/metadata/glycan_binders_custom.tsv')

    if per_head_mode:
        td_k = _prep_data_for_head('k', exp_cfg, pl_k, assoc_map, glycan,
                                    experiment_dir)
        td_o = _prep_data_for_head('o', exp_cfg, pl_o, assoc_map, glycan,
                                    experiment_dir)
        print(f'  K dataset: {len(td_k.md5_list)} MD5s, '
              f'{len(td_k.k_classes)} K classes')
        print(f'  O dataset: {len(td_o.md5_list)} MD5s, '
              f'{len(td_o.o_classes)} O classes')
    else:
        training_config = TrainingConfig.from_dict(exp_cfg)
        td = prepare_training_data(training_config, assoc_map, glycan)
        td.save(experiment_dir)
        td_k = td_o = td

    md5_to_idx_k = {m: i for i, m in enumerate(td_k.md5_list)}
    md5_to_idx_o = {m: i for i, m in enumerate(td_o.md5_list)}

    print('\nStep 2: Creating independent K and O splits...')

    set_seed(seed)
    k_md5s = _md5s_with_labels(td_k.k_labels, td_k.md5_list)
    primary_k = [td_k.k_classes[int(td_k.k_labels[md5_to_idx_k[m]].argmax())]
                 for m in k_md5s]
    splits_k = create_stratified_split(
        k_md5s, primary_k,
        train_ratio=train_cfg.get('train_ratio', 0.7),
        val_ratio=train_cfg.get('val_ratio', 0.15),
        seed=seed,
    )
    print(f'  K split:  Train: {len(splits_k["train"])}, '
          f'Val: {len(splits_k["val"])}, '
          f'Test: {len(splits_k["test"])} '
          f'(of {len(k_md5s)} K-labeled MD5s)')

    set_seed(seed + 1)
    o_md5s = _md5s_with_labels(td_o.o_labels, td_o.md5_list)
    primary_o = [td_o.o_classes[int(td_o.o_labels[md5_to_idx_o[m]].argmax())]
                 for m in o_md5s]
    splits_o = create_stratified_split(
        o_md5s, primary_o,
        train_ratio=train_cfg.get('train_ratio', 0.7),
        val_ratio=train_cfg.get('val_ratio', 0.15),
        seed=seed + 1,
    )
    print(f'  O split:  Train: {len(splits_o["train"])}, '
          f'Val: {len(splits_o["val"])}, '
          f'Test: {len(splits_o["test"])} '
          f'(of {len(o_md5s)} O-labeled MD5s)')

    with open(os.path.join(experiment_dir, 'splits_k.json'), 'w') as f:
        json.dump(splits_k, f, indent=2)
    with open(os.path.join(experiment_dir, 'splits_o.json'), 'w') as f:
        json.dump(splits_o, f, indent=2)

    print('\nStep 3: Loading per-residue embeddings...')
    emb_file = data_cfg.get(
        'embedding_file',
        'data/training_data/embeddings/esm2_650m_full_md5.npz')
    # Load the union -- both heads use the same emb_dict but only their
    # own MD5s actually get touched.
    md5_set = set(td_k.md5_list) | set(td_o.md5_list)
    emb_dict = load_embeddings_or_bins(emb_file, md5_filter=md5_set)
    print(f'  Loaded {len(emb_dict)} embeddings from {emb_file}')

    min_coverage = float(train_cfg.get('min_embedding_coverage', 0.5))

    def _build_arrays(split_md5s, labels, md5_to_idx, split_name):
        valid = check_embedding_coverage(
            split_md5s, emb_dict, min_coverage, split_name, emb_file)
        valid = [m for m in valid if m in md5_to_idx]
        X = [emb_dict[m] for m in valid]
        idxs = [md5_to_idx[m] for m in valid]
        y = labels[idxs]
        return X, y

    print('\nStep 4: Training K head...')
    set_seed(seed)
    X_train_k, y_train_k = _build_arrays(splits_k['train'], td_k.k_labels,
                                          md5_to_idx_k, 'k/train')
    X_val_k, y_val_k = _build_arrays(splits_k['val'], td_k.k_labels,
                                      md5_to_idx_k, 'k/val')
    X_test_k, y_test_k = _build_arrays(splits_k['test'], td_k.k_labels,
                                        md5_to_idx_k, 'k/test')
    train_head('k', X_train_k, y_train_k, X_val_k, y_val_k, X_test_k, y_test_k,
               td_k.k_classes, config, os.path.join(experiment_dir, 'model_k'))

    print('\nStep 5: Training O head...')
    set_seed(seed)
    X_train_o, y_train_o = _build_arrays(splits_o['train'], td_o.o_labels,
                                          md5_to_idx_o, 'o/train')
    X_val_o, y_val_o = _build_arrays(splits_o['val'], td_o.o_labels,
                                      md5_to_idx_o, 'o/val')
    X_test_o, y_test_o = _build_arrays(splits_o['test'], td_o.o_labels,
                                        md5_to_idx_o, 'o/test')
    train_head('o', X_train_o, y_train_o, X_val_o, y_val_o, X_test_o, y_test_o,
               td_o.o_classes, config, os.path.join(experiment_dir, 'model_o'))

    from cipher.provenance import capture_provenance
    experiment_meta = {
        'model': 'light_attention_binary',
        'config': config,
        'filter_mode': 'per_head' if per_head_mode else 'single_list',
        'positive_list_k_path': pl_k,
        'positive_list_o_path': pl_o,
        'n_md5s_k': len(td_k.md5_list),
        'n_md5s_o': len(td_o.md5_list),
        'n_k_classes': len(td_k.k_classes),
        'n_o_classes': len(td_o.o_classes),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'provenance': capture_provenance(),
    }
    with open(os.path.join(experiment_dir, 'experiment.json'), 'w') as f:
        json.dump(experiment_meta, f, indent=2)

    print(f'\nTraining complete. Results in {experiment_dir}')
