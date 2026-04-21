# Contrastive Encoder (ArcFace + class-balanced + cluster-stratified)

**Status:** scaffolding — sampler.py implemented and tested; model.py + train.py to follow.

## Why

The PHL label audit showed ESM-2 mean embeddings place proteins of different K-types
into the same region of feature space — only 11% of PHL proteins share a K-type with
their nearest training neighbour, vs 2.9% random baseline. The prior SupCon attempt
(in the old klebsiella repo) did not fix this for two reasons:

1. Random batch sampling was dominated by majority K-types (K1: 1000 proteins,
   rare K-types like KL151: 450 proteins) → gradient weak for rare classes.
2. SupCon provides only implicit margin; with baseline cosines already at 0.998,
   gradients flatten.

This module trains an encoder on top of ESM-2 (mean or per-residue) that projects
into a space where same-K-type proteins cluster explicitly.

## Architecture (planned)

### Encoder backbone (`model.py`, TODO)

```
embedding (D_in)
    → Linear → BatchNorm → ReLU → Dropout     (x3, dims 1280 -> 1024 -> 1024)
    → Linear → L2-normalize
    → representation z (1280-d, drop-in replacement for ESM-2)
```

### Loss head (`model.py`, TODO)

**ArcFace** (additive angular margin softmax, Deng et al. 2019). For each class k:

- learned centroid w_k on the unit hypersphere
- logit = cos(theta_k + m) for the ground-truth class; cos(theta_k) otherwise
- typical margin m = 0.5 rad, scale s = 30

Applied independently on K-type and O-type heads, shared backbone, losses summed
with configurable weights (`lambda_k`, `lambda_o`).

### Sampler (`sampler.py`, ✅ implemented)

**PK + cluster-stratified batches**:
1. Each batch samples P K-types uniformly from usable classes (classes with
   at least K cluster-distinct samples after the `prepare_training_data` pipeline).
2. Per chosen K-type, round-robin across its 70%-identity clusters until K
   samples are drawn.

Net effect: batch = P × K samples where every K-type contributes equally
regardless of training frequency, and the K samples per class are diverse (1 per
cluster until exhausted).

Default: P = 32, K = 8, batch = 256. Unit tests in `tests/test_contrastive_sampler.py`.

### Training loop (`train.py`, TODO)

1. Reuse `cipher.data.prepare_training_data` for filtering/downsampling/clustering.
2. Build `PKClusterSampler` from the resulting training tensor.
3. Forward pass → ArcFace loss (K + O) → backward, optimizer step.
4. Validate: compute mean intra-class vs inter-class cosine on a held-out split
   (K-type match rate proxy) every N epochs — early stop on that, not on loss.
5. At training end:
   - Save `encoder.pt`
   - Run the encoder over the full training NPZ + validation NPZ → write
     `contrastive_train_md5.npz` and `contrastive_val_md5.npz` in the run dir.
6. Print instructions to train a downstream `attention_mlp` on the generated NPZs.

## Workflow

```bash
# 1. Train the encoder
cipher-train --model contrastive_encoder \
    --embedding_file .../esm2_650m_md5.npz \
    --val_embedding_file .../validation_esm2_650m_md5.npz \
    --positive_list .../pipeline_positive.list \
    --cluster_file .../candidates_clusters.tsv --cluster_threshold 70 \
    --name posList_cl70_arcface_k_o

# -> produces: experiments/contrastive_encoder/posList_cl70_arcface_k_o/
#    encoder.pt + contrastive_train_md5.npz + contrastive_val_md5.npz

# 2. Train a downstream classifier on the learned embeddings
cipher-train --model attention_mlp \
    --embedding_file experiments/contrastive_encoder/posList_cl70_arcface_k_o/contrastive_train_md5.npz \
    --val_embedding_file experiments/contrastive_encoder/posList_cl70_arcface_k_o/contrastive_val_md5.npz \
    --positive_list .../pipeline_positive.list \
    --cluster_file .../candidates_clusters.tsv --cluster_threshold 70 \
    --name downstream_on_contrastive

# 3. Evaluate as usual
cipher-evaluate experiments/attention_mlp/downstream_on_contrastive/
```

## Quality gate (recommended)

Immediately after step 1 finishes, run the PHL neighbour-label audit on the new
NPZs:

```bash
python scripts/analysis/phl_neighbor_labels.py \
    --train-emb experiments/contrastive_encoder/posList_cl70_arcface_k_o/contrastive_train_md5.npz \
    --val-emb  experiments/contrastive_encoder/posList_cl70_arcface_k_o/contrastive_val_md5.npz \
    --out-dir  experiments/contrastive_encoder/posList_cl70_arcface_k_o/analysis \
    --restrict-to-labeled
```

Target: top-1 K-match rate > 20% (vs 11.3% baseline on raw ESM-2 650M mean).
If it doesn't move, the architecture needs revisiting before burning downstream
training runs.

## Files

| File | Status | Purpose |
|---|---|---|
| `base_config.yaml` | TODO | Default hyperparameters |
| `model.py` | TODO | Encoder backbone + ArcFace heads |
| `sampler.py` | ✅ | PK + cluster-stratified batch sampler |
| `train.py` | TODO | Training loop + NPZ generation |
| `predict.py` | TODO | Stub for cipher-evaluate compat (raises with instructions to train downstream classifier) |
| `README.md` | ✅ | This file |
