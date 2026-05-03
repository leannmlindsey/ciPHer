# multi_tower_attention_mlp

Three-tower mid-fusion architecture for K/O serotype prediction.
Each input embedding type has its own SE-attention + MLP tower; the
three tower latents are concatenated and fed to a shared K head + O
head.

## Why mid-fusion

From agent 1's 2026-05-03 diagnostic on failed concat experiments
(see `notes/inbox/agent3/2026-05-03-1900-from-agent1-multi-tower-design-handoff.md`):

- `sweep_kmer_aa20_k4` alone: PHL OR HR@1 = 0.560
- `sweep_attention_3b_mean` alone: ~0.45
- `concat_prott5_mean+kmer_aa20_k4`: 0.34 (-22pp vs kmer alone)
- `concat_esm2_3b_mean+kmer_aa20_k4`: 0.33 (-23pp vs kmer alone)
- Best hybrid (cross-model OR): **0.620 PHL / 0.800 overall**

Diagnosis: in plain concat, the sparse 160k-d kmer block gets ignored
in the shared first Linear layer because the dense pLM block dominates
gradient magnitudes. The hybrid result proves the three signals are
genuinely complementary; we just need an architecture that keeps them
from trampling each other in the early layers. Mid-fusion (each tower
projects to a 256-d latent FIRST, then concat) handles this.

## Architecture

```
ProtT5 (1024-d)     -> Tower A: SE-attn + MLP -> 256-d latent
ESM-2 3B (2560-d)   -> Tower B: SE-attn + MLP -> 256-d latent
kmer aa20 k4 (160k) -> Tower C: SE-attn + MLP -> 256-d latent
                                                     |
                              concat 3 x 256 -> 768-d
                                                /        \
                                          K head         O head
```

## Status

Scaffolding only. Build is queued behind the current
light_attention_binary v3_uat run (job 2236031); pick up when LA wraps.

## Bar to beat

- Best single model: **0.560** PHL OR (sweep_kmer_aa20_k4)
- Best hybrid:        **0.620** PHL OR (esm2_3b_K + kmer_aa20_O)
- Multi-tower goal:   beat 0.620

## Reference

`notes/inbox/agent3/2026-05-03-1900-from-agent1-multi-tower-design-handoff.md`
for the full design spec including locked defaults (warm-start sources,
tower latent dim, training-data alignment, optimizer, loss).
