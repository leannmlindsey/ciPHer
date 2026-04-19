"""Training script for AttentionMLP serotype prediction model.

Trains separate K-type and O-type heads using the same embeddings.
Called by cipher-train CLI or directly.
"""

import json
import os
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from cipher.data.training import TrainingConfig, prepare_training_data
from cipher.data.splits import create_stratified_split
from cipher.data.embeddings import load_embeddings

from model import AttentionMLP, SerotypeDataset, get_loss_fn, compute_metrics


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, loader, strategy, device):
    """Run evaluation on a data loader."""
    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for batch_emb, batch_labels in loader:
            batch_emb = batch_emb.to(device)
            logits = model(batch_emb)
            all_logits.append(logits.cpu())
            all_labels.append(batch_labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    return compute_metrics(all_logits, all_labels, strategy)


def train_head(head_name, X_train, y_train, X_val, y_val, X_test, y_test,
               classes, config, output_dir):
    """Train one head (K or O) and save results.

    Args:
        head_name: 'k' or 'o'
        X_train/val/test: numpy arrays (N, embedding_dim)
        y_train/val/test: numpy arrays (N, n_classes)
        classes: list of class names
        config: full experiment config dict
        output_dir: path to save model_k/ or model_o/

    Returns:
        training history dict
    """
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback: simple range wrapper
        def tqdm(iterable, **kwargs):
            return iterable

    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})
    strategy = config.get('experiment', {}).get('label_strategy', 'single_label')

    hidden_dims = model_cfg.get('hidden_dims', [1280, 640, 320, 160])
    se_dim = model_cfg.get('attention_dim', 640)
    dropout = model_cfg.get('dropout', 0.1)
    lr = train_cfg.get('learning_rate', 1e-5)
    epochs = train_cfg.get('epochs', 200)
    patience = train_cfg.get('patience', 30)
    batch_size = train_cfg.get('batch_size', 64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n  Training {head_name.upper()} head ({len(classes)} classes)')
    print(f'  Device: {device}')
    print(f'  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}')

    # Create model
    input_dim = X_train.shape[1]
    model = AttentionMLP(
        input_dim=input_dim,
        num_classes=len(classes),
        hidden_dims=hidden_dims,
        se_dim=se_dim,
        dropout=dropout,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Parameters: {trainable_params:,} trainable / {total_params:,} total')

    # Create data loaders
    train_dataset = SerotypeDataset(X_train, y_train)
    val_dataset = SerotypeDataset(X_val, y_val)
    test_dataset = SerotypeDataset(X_test, y_test)

    use_pin = torch.cuda.is_available()
    n_workers = 4 if torch.cuda.is_available() else 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=n_workers, pin_memory=use_pin)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False,
                            num_workers=n_workers, pin_memory=use_pin)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=False,
                             num_workers=n_workers, pin_memory=use_pin)

    # Compute per-class pos_weight for BCE losses on imbalanced multi-label
    # data. Without this, BCEWithLogitsLoss on sparse labels collapses to
    # "predict zero for everything" since negatives massively outnumber positives.
    # Formula: pos_weight[i] = num_negatives_i / num_positives_i, capped at 100
    # to prevent gradient explosion for ultra-rare classes.
    use_pos_weight = config.get('training', {}).get('use_pos_weight', True)
    pos_weight = None
    if use_pos_weight and strategy in ('multi_label', 'multi_label_threshold',
                                         'weighted_multi_label'):
        # For weighted_multi_label, treat any nonzero target as a positive
        binary = (y_train > 0).astype(np.float64)
        n_samples = binary.shape[0]
        pos_counts = binary.sum(axis=0)
        neg_counts = n_samples - pos_counts
        pos_counts = np.maximum(pos_counts, 1.0)
        pw = np.minimum(neg_counts / pos_counts, 100.0)
        pos_weight = torch.tensor(pw, dtype=torch.float32).to(device)
        print(f'  pos_weight: range [{pw.min():.1f}, {pw.max():.1f}], mean={pw.mean():.1f}')

    # Loss, optimizer, scheduler
    loss_fn = get_loss_fn(strategy, class_weights=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)

    # Training loop
    os.makedirs(output_dir, exist_ok=True)
    best_val_metric = -float('inf')
    patience_counter = 0
    history = {'train': [], 'val': [], 'test': []}

    epoch_iter = tqdm(range(epochs), desc=f'  {head_name.upper()} head', unit='epoch')
    for epoch in epoch_iter:
        model.train()
        train_loss = 0
        n_batches = 0

        for batch_emb, batch_labels in train_loader:
            batch_emb = batch_emb.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_emb)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = train_loss / max(n_batches, 1)

        # Evaluate
        val_metrics = evaluate_model(model, val_loader, strategy, device)
        test_metrics = evaluate_model(model, test_loader, strategy, device)

        history['train'].append({'loss': avg_loss})
        history['val'].append(val_metrics)
        history['test'].append(test_metrics)

        # Early stopping metric and progress label depend on strategy
        if strategy == 'single_label':
            current_metric = val_metrics['top1_acc']
            metric_label = 'prot_sero_acc'
        elif strategy in ('multi_label', 'multi_label_threshold',
                          'weighted_multi_label'):
            current_metric = val_metrics['f1_macro']
            metric_label = 'F1_macro'
        elif strategy == 'weighted_soft':
            current_metric = -val_metrics['kl_div']
            metric_label = 'neg_KL'
        else:
            current_metric = val_metrics.get('top1_acc', 0)
            metric_label = 'metric'

        if hasattr(epoch_iter, 'set_postfix'):
            postfix = {
                'loss': f'{avg_loss:.4f}',
                metric_label: f'{current_metric:.4f}',
                'best': f'{best_val_metric:.4f}' if best_val_metric > -float('inf') else '-',
                'pat': patience_counter,
            }
            # Show both micro and macro F1 for multi-label strategies
            if 'f1_micro' in val_metrics and 'f1_macro' in val_metrics:
                postfix['F1_micro'] = f'{val_metrics["f1_micro"]:.4f}'
            epoch_iter.set_postfix(postfix)

        if current_metric > best_val_metric:
            best_val_metric = current_metric
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'  Early stopping at epoch {epoch + 1}')
            break

    # Save config for this head
    head_config = {
        'head': head_name,
        'strategy': strategy,
        'input_dim': input_dim,
        'num_classes': len(classes),
        'classes': classes,
        'hidden_dims': hidden_dims,
        'se_dim': se_dim,
        'dropout': dropout,
        'lr': lr,
        'weight_decay': 0.01,
        'epochs': epochs,
        'patience': patience,
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

    # Print final metrics
    best_epoch = epoch + 1 - patience_counter
    best_val = history['val'][best_epoch - 1]
    best_test = history['test'][best_epoch - 1]
    print(f'  Best epoch: {best_epoch}')
    if strategy == 'single_label':
        print(f'  Val  protein-serotype acc: {best_val.get("top1_acc", 0):.4f}')
        print(f'  Test protein-serotype acc: {best_test.get("top1_acc", 0):.4f}')
    elif strategy in ('multi_label', 'multi_label_threshold',
                      'weighted_multi_label'):
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
    elif strategy == 'weighted_soft':
        print(f'  Val  KL: {best_val.get("kl_div", 0):.4f}  '
              f'cos_sim: {best_val.get("cos_sim", 0):.4f}  '
              f'top1_in_pos: {best_val.get("top1_acc", 0):.4f}')
        print(f'  Test KL: {best_test.get("kl_div", 0):.4f}  '
              f'cos_sim: {best_test.get("cos_sim", 0):.4f}  '
              f'top1_in_pos: {best_test.get("top1_acc", 0):.4f}')

    return history


def train(experiment_dir, config):
    """Train both K and O heads for the AttentionMLP model.

    Args:
        experiment_dir: path to experiment directory (will contain model_k/, model_o/)
        config: full experiment config dict
    """
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    exp_cfg = config.get('experiment', {})
    seed = train_cfg.get('seed', 42)

    print('=' * 70)
    print('TRAINING AttentionMLP')
    print('=' * 70)
    print(f'  Experiment dir: {experiment_dir}')
    print(f'  Seed: {seed}')

    # Step 1: Prepare training data
    print('\nStep 1: Preparing training data...')
    training_config = TrainingConfig.from_dict(exp_cfg)
    td = prepare_training_data(
        training_config,
        data_cfg.get('association_map',
                     'data/training_data/metadata/host_phage_protein_map.tsv'),
        data_cfg.get('glycan_binders',
                     'data/training_data/metadata/glycan_binders_custom.tsv'),
    )

    # Save training data summary
    td.save(experiment_dir)

    # Step 2: Create train/val/test splits — INDEPENDENT for K and O.
    # K head uses a K-stratified split over MD5s with K labels.
    # O head uses an O-stratified split over MD5s with O labels.
    # This ensures each head gets balanced rare-class representation.
    print('\nStep 2: Creating independent K and O splits...')

    def md5s_with_labels(labels, md5_list):
        """Return MD5s that have at least one nonzero label."""
        keep = (labels > 0).any(axis=1)
        return [m for m, k in zip(md5_list, keep) if k]

    def primary_class(md5, labels, md5_to_idx, classes):
        return classes[int(labels[md5_to_idx[m]].argmax())]

    md5_to_idx_full = {m: i for i, m in enumerate(td.md5_list)}

    # K split
    set_seed(seed)
    k_md5s = md5s_with_labels(td.k_labels, td.md5_list)
    primary_k = [td.k_classes[int(td.k_labels[md5_to_idx_full[m]].argmax())]
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

    # O split (use seed+1 so the splits are not artificially correlated)
    set_seed(seed + 1)
    o_md5s = md5s_with_labels(td.o_labels, td.md5_list)
    primary_o = [td.o_classes[int(td.o_labels[md5_to_idx_full[m]].argmax())]
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

    # Save both splits
    with open(os.path.join(experiment_dir, 'splits_k.json'), 'w') as f:
        json.dump(splits_k, f, indent=2)
    with open(os.path.join(experiment_dir, 'splits_o.json'), 'w') as f:
        json.dump(splits_o, f, indent=2)

    # Step 3: Load embeddings (only those needed for either split)
    print('\nStep 3: Loading embeddings...')
    emb_file = data_cfg.get('embedding_file',
                            'data/training_data/embeddings/esm2_650m_md5.npz')
    md5_set = set(td.md5_list)
    emb_dict = load_embeddings(emb_file, md5_filter=md5_set)
    print(f'  Loaded {len(emb_dict)} embeddings')

    # Step 4: Build arrays
    md5_to_idx = md5_to_idx_full

    def build_arrays(split_md5s, labels):
        valid = [m for m in split_md5s if m in emb_dict and m in md5_to_idx]
        X = np.array([emb_dict[m] for m in valid])
        idxs = [md5_to_idx[m] for m in valid]
        y = labels[idxs]
        return X, y

    # Step 5: Train K head
    print('\nStep 4: Training...')
    set_seed(seed)

    X_train_k, y_train_k = build_arrays(splits_k['train'], td.k_labels)
    X_val_k, y_val_k = build_arrays(splits_k['val'], td.k_labels)
    X_test_k, y_test_k = build_arrays(splits_k['test'], td.k_labels)

    train_head('k', X_train_k, y_train_k, X_val_k, y_val_k, X_test_k, y_test_k,
               td.k_classes, config, os.path.join(experiment_dir, 'model_k'))

    # Step 6: Train O head
    set_seed(seed)

    X_train_o, y_train_o = build_arrays(splits_o['train'], td.o_labels)
    X_val_o, y_val_o = build_arrays(splits_o['val'], td.o_labels)
    X_test_o, y_test_o = build_arrays(splits_o['test'], td.o_labels)

    train_head('o', X_train_o, y_train_o, X_val_o, y_val_o, X_test_o, y_test_o,
               td.o_classes, config, os.path.join(experiment_dir, 'model_o'))

    # Save experiment metadata
    experiment_meta = {
        'model': 'attention_mlp',
        'config': config,
        'n_md5s': len(td.md5_list),
        'n_k_classes': len(td.k_classes),
        'n_o_classes': len(td.o_classes),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(experiment_dir, 'experiment.json'), 'w') as f:
        json.dump(experiment_meta, f, indent=2)

    print(f'\nTraining complete. Results in {experiment_dir}')
