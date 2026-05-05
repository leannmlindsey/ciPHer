"""ciphermil training — bag-level MIL on phage→host serotype prediction.

Training pipeline:
  1. Run cipher.data.training.prepare_training_data to apply all the
     standard cipher filters (tools / positive_list / cluster_threshold /
     min_class_samples / min_sources / etc.) and get the filtered MD5
     set.
  2. Re-walk the association map to group MD5s into PHAGE bags. Each
     phage's bag = the MD5s in the filtered set that appear in the
     phage's protein list.
  3. Compute single-label per phage = MODE of host K-types (or O-types,
     depending on which head is being trained). Drop phages with
     ambiguous mode (no clear plurality).
  4. Phage-level stratified train/val/test split (NOT protein-level).
  5. Train one MIL model per requested head. Per leann's 2026-05-05
     design call: K and O models are SEPARATE (no shared trunk), so
     each head loops independently.

Loss: CrossEntropyLoss over the bag-level logits (single-label softmax).

Note on the embedding lookup: the standard cipher pipeline keys
embeddings by MD5 (deduplicates identical sequences). For MIL bag
aggregation, the SAME md5 may appear in MULTIPLE phages. That's fine —
the bag for phage A may contain md5_x at one position; the bag for
phage B may also contain md5_x; both retrieve the same vector.
"""

import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

# Add this model's dir to sys.path for `import model` (cipher convention).
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cipher.data.training import TrainingConfig, prepare_training_data
from cipher.data.embeddings import load_embeddings, load_embeddings_concat
from cipher.data.proteins import load_fasta_md5
from model import AttentionMIL  # local import


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_association_map(path):
    """Yield (phage_id, protein_id, host_K, host_O, md5) tuples from
    host_phage_protein_map.tsv."""
    import csv
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            # Tolerate either header convention.
            phage = row.get('phage_id') or row.get('phage_genome', '')
            protein = row.get('protein_id', '')
            host_k = (row.get('host_K') or row.get('K_type') or '').strip()
            host_o = (row.get('host_O') or row.get('O_type') or '').strip()
            md5 = (row.get('protein_md5') or row.get('md5') or '').strip()
            yield phage, protein, host_k, host_o, md5


def _build_phage_bags(association_map_path, kept_md5s, head, classes):
    """Group MD5s by phage and assign single-label (mode) per phage.

    Args:
        association_map_path: host_phage_protein_map.tsv
        kept_md5s: set of md5s that survived prepare_training_data's filters
        head: 'k' or 'o'
        classes: list of class names (K-types or O-types) the model can
            predict. Phages whose mode label is not in this list are
            dropped (out-of-vocabulary at training time).

    Returns:
        list of (phage_id, list_of_md5s, label_idx) tuples, one per phage
        with non-ambiguous label. Plus a summary dict for logging.
    """
    label_field = 'host_K' if head == 'k' else 'host_O'
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # phage_id -> list of (md5, host_K_or_O) for proteins of this phage in kept_md5s
    phage_md5s = defaultdict(set)         # phage -> {md5, ...}  (deduped)
    phage_labels = defaultdict(list)      # phage -> [host_label, ...]  (one per association)

    for phage, protein, host_k, host_o, md5 in _load_association_map(association_map_path):
        if not phage or not md5:
            continue
        if md5 not in kept_md5s:
            continue
        host_label = host_k if head == 'k' else host_o
        if not host_label or host_label.lower() == 'null':
            continue
        phage_md5s[phage].add(md5)
        phage_labels[phage].append(host_label)

    bags = []
    n_no_clear_mode = 0
    n_oov_label = 0
    n_dropped_empty = 0
    for phage, md5_set in phage_md5s.items():
        labels = phage_labels[phage]
        if not labels:
            n_dropped_empty += 1
            continue
        # Mode of labels. If multiple modes (tied plurality), drop.
        counter = Counter(labels)
        most_common = counter.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            n_no_clear_mode += 1
            continue
        mode_label = most_common[0][0]
        if mode_label not in class_to_idx:
            n_oov_label += 1
            continue
        bags.append((phage, sorted(md5_set), class_to_idx[mode_label]))

    summary = {
        'n_phages_input': len(phage_md5s),
        'n_phages_kept': len(bags),
        'n_dropped_empty_label': n_dropped_empty,
        'n_dropped_no_clear_mode': n_no_clear_mode,
        'n_dropped_oov_label': n_oov_label,
    }
    return bags, summary


def _split_bags_stratified(bags, train_ratio, val_ratio, seed):
    """Stratified split of phage bags by class label.

    For each class, shuffle that class's bags and assign train_ratio to
    train, val_ratio to val, the rest to test.
    """
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for b in bags:
        by_class[b[2]].append(b)

    train, val, test = [], [], []
    for cls, cls_bags in by_class.items():
        cls_bags = list(cls_bags)
        rng.shuffle(cls_bags)
        n = len(cls_bags)
        n_train = max(1, int(round(n * train_ratio))) if n > 1 else n
        n_val = max(0, int(round(n * val_ratio))) if n > 2 else 0
        train.extend(cls_bags[:n_train])
        val.extend(cls_bags[n_train:n_train + n_val])
        test.extend(cls_bags[n_train + n_val:])
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def _bag_to_tensor(bag, emb_dict, device):
    """Look up each MD5 in emb_dict; stack to (n, input_dim) tensor."""
    embs = [emb_dict[m] for m in bag if m in emb_dict]
    if not embs:
        return None
    return torch.tensor(np.stack(embs), dtype=torch.float32, device=device)


def _evaluate(model, bags, emb_dict, device, num_classes):
    """Compute val/test top-1 accuracy + cross-entropy on a set of bags."""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for phage, md5s, label_idx in bags:
            x = _bag_to_tensor(md5s, emb_dict, device)
            if x is None:
                continue
            logits, _ = model(x)
            label = torch.tensor([label_idx], device=device)
            loss_sum += loss_fn(logits, label).item()
            pred = int(torch.argmax(logits, dim=1).item())
            correct += int(pred == label_idx)
            total += 1
    return {
        'top1_acc': correct / max(total, 1),
        'loss': loss_sum / max(total, 1),
        'n': total,
    }


def train_head(head, td, emb_dict, config, output_dir, classes):
    """Train one MIL head (K or O). Saves best_model.pt + config.json.

    Args:
        head: 'k' or 'o'
        td: TrainingData from prepare_training_data
        emb_dict: dict {md5: embedding_array}
        config: full experiment config
        output_dir: where to save model_k/ or model_o/
        classes: list of class names for this head
    """
    model_cfg = config.get('model', {})
    train_cfg = config.get('training', {})

    L = model_cfg.get('L', 800)
    D = model_cfg.get('D', 128)
    K = model_cfg.get('K', 1)
    dropout = model_cfg.get('dropout', 0.1)
    lr = train_cfg.get('learning_rate', 5e-4)
    epochs = train_cfg.get('epochs', 150)
    patience = train_cfg.get('patience', 30)
    weight_decay = train_cfg.get('weight_decay', 1e-4)
    seed = train_cfg.get('seed', 42)
    train_ratio = train_cfg.get('train_ratio', 0.7)
    val_ratio = train_cfg.get('val_ratio', 0.15)

    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build phage bags from the filtered MD5 set.
    kept_md5s = set(td.md5_list)
    bags, bag_summary = _build_phage_bags(
        config['data']['association_map'], kept_md5s, head, classes)
    print(f'\n  Building {head.upper()} bags from {len(kept_md5s)} kept MD5s ...')
    for k, v in bag_summary.items():
        print(f'    {k}: {v}')
    if not bags:
        print(f'  ERROR: no bags survived. Skipping {head} head.')
        return None

    train_bags, val_bags, test_bags = _split_bags_stratified(
        bags, train_ratio, val_ratio, seed)
    print(f'  Bags split — train: {len(train_bags)}, val: {len(val_bags)}, test: {len(test_bags)}')

    # Determine input dim from a sample embedding.
    sample_md5 = next(iter(emb_dict))
    input_dim = emb_dict[sample_md5].shape[-1]

    model = AttentionMIL(
        input_dim=input_dim,
        num_classes=len(classes),
        L=L, D=D, K=K, dropout=dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model: AttentionMIL  input_dim={input_dim}  L={L}  D={D}  '
          f'n_classes={len(classes)}  params={n_params:,}')

    optimizer = optim.Adam(model.parameters(), lr=lr,
                           betas=(0.9, 0.999), weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6)
    loss_fn = torch.nn.CrossEntropyLoss()

    os.makedirs(output_dir, exist_ok=True)
    best_val_acc = -1.0
    patience_counter = 0
    history = {'train': [], 'val': [], 'test': []}

    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(train_bags)

        epoch_loss = 0.0
        n_seen = 0
        for phage, md5s, label_idx in train_bags:
            x = _bag_to_tensor(md5s, emb_dict, device)
            if x is None:
                continue
            label = torch.tensor([label_idx], device=device)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = loss_fn(logits, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
            n_seen += 1
        scheduler.step()

        avg_loss = epoch_loss / max(n_seen, 1)

        val_metrics = _evaluate(model, val_bags, emb_dict, device, len(classes))
        test_metrics = _evaluate(model, test_bags, emb_dict, device, len(classes))

        history['train'].append({'loss': avg_loss, 'n': n_seen})
        history['val'].append(val_metrics)
        history['test'].append(test_metrics)

        if val_metrics['top1_acc'] > best_val_acc:
            best_val_acc = val_metrics['top1_acc']
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pt'))
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == 1 or epoch >= epochs - 5:
            print(f'  epoch {epoch:4d}/{epochs}  '
                  f'train_loss={avg_loss:.4f}  '
                  f'val_top1={val_metrics["top1_acc"]:.4f}  '
                  f'(best={best_val_acc:.4f}, pat={patience_counter})  '
                  f'test_top1={test_metrics["top1_acc"]:.4f}')

        if patience_counter >= patience:
            print(f'  Early stopping at epoch {epoch} (best val_top1={best_val_acc:.4f})')
            break

    # Save head config
    head_config = {
        'head': head,
        'input_dim': input_dim,
        'num_classes': len(classes),
        'classes': classes,
        'L': L, 'D': D, 'K': K,
        'dropout': dropout,
        'lr': lr, 'weight_decay': weight_decay,
        'epochs': epochs, 'patience': patience,
        'train_ratio': train_ratio, 'val_ratio': val_ratio,
        'seed': seed,
        'n_train_bags': len(train_bags),
        'n_val_bags': len(val_bags),
        'n_test_bags': len(test_bags),
        'best_val_top1_acc': best_val_acc,
        'bag_summary': bag_summary,
    }
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(head_config, f, indent=2)
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    print(f'  {head.upper()} head complete. Best val top1: {best_val_acc:.4f}')
    print(f'  Saved: {output_dir}/best_model.pt + config.json + history.json')
    return head_config


def train(experiment_dir, config):
    """Train both K and O heads (or whichever requested via training.heads).

    Per cipher.cli.train_runner convention: this is the entry point.
    Saves model_k/ and/or model_o/ subdirs under experiment_dir.
    """
    print('=' * 60)
    print('TRAINING ciphermil (attention-based MIL, single-label softmax)')
    print('=' * 60)

    # Load filtered training data via the standard cipher pipeline.
    print('\nStep 1: prepare_training_data ...')
    tc = TrainingConfig.from_dict(config.get('experiment', {}))
    td = prepare_training_data(
        tc,
        config['data']['association_map'],
        config['data']['glycan_binders'],
        verbose=True,
    )

    # Load embeddings for the filtered MD5 set.
    print('\nStep 2: load embeddings ...')
    emb_path = config['data']['embedding_file']
    emb_path_2 = config.get('data', {}).get('embedding_file_2')
    needed = set(td.md5_list)
    if emb_path_2:
        emb_dict = load_embeddings_concat(emb_path, emb_path_2, md5_filter=needed)
    else:
        emb_dict = load_embeddings(emb_path)
    print(f'  Loaded {len(emb_dict)} embeddings ({sum(m in emb_dict for m in needed)} of {len(needed)} kept MD5s)')

    # Decide which heads to train.
    heads = config.get('training', {}).get('heads', 'both')
    if heads == 'k':
        head_list = ['k']
    elif heads == 'o':
        head_list = ['o']
    else:
        head_list = ['k', 'o']

    print(f'\nStep 3: training heads — {head_list}')
    summaries = {}
    for h in head_list:
        out_dir = os.path.join(experiment_dir, f'model_{h}')
        classes = td.k_classes if h == 'k' else td.o_classes
        print(f'\n--- Training {h.upper()} head ({len(classes)} classes) ---')
        s = train_head(h, td, emb_dict, config, out_dir, classes)
        if s is not None:
            summaries[h] = s

    # Save experiment-level summary
    exp_meta = {
        'model': 'ciphermil',
        'config': config,
        'heads_trained': list(summaries.keys()),
        'k_summary': summaries.get('k'),
        'o_summary': summaries.get('o'),
        'finished_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(experiment_dir, 'experiment.json'), 'w') as f:
        json.dump(exp_meta, f, indent=2)
    print(f'\nDone. Trained heads: {list(summaries.keys())}')
    print(f'Experiment metadata: {experiment_dir}/experiment.json')
