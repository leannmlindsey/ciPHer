# PHL ceiling analysis — findings (2026-04-20 late)

## Question
Is PHL hard because its proteins are distant from training (distribution shift), or hard despite having close training neighbours (modelling / representation limit)?

## Part 1 — Are PHL proteins close to anything in training?

Yes, very.

| Metric | PHL → training | Within-training nearest-other (sample n=500) |
|---|---:|---:|
| Mean nearest cosine | **0.997** | 0.995 |
| Median nearest cosine | **0.998** | 0.998 |
| Min PHL nearest | 0.976 | — |

- 90.6% of PHL proteins have some training protein at cosine ≥ 0.99.
- 100% are ≥ 0.95.
- Zero exact MD5 matches — PHL proteins are not literally in training, but have near-identical embeddings.

**Conclusion:** PHL is **not distribution-shifted in ESM-2 embedding space.** The prior 3-mer Jaccard ~0.14 story is real at the raw sequence level but ESM-2 has collapsed that difference.

## Part 2 — Does the nearest neighbour share a K-type?

No — much worse than expected.

| Measure | Fraction | ×above random |
|---|---:|---:|
| Random baseline (random PHL vs random labelled training protein) | 2.9% | 1× |
| Top-1 nearest neighbour shares a K-type | **10.8%** | 3.7× |
| Top-5 has at least one match | 21.6% | 7.4× |
| Top-10 has at least one match | 26.1% | 9.0× |
| **No match anywhere in top-50** | **55.9%** | — |

**Conclusion:** ESM-2 mean embeddings do carry some K-type signal (3.7× above chance at top-1), but it's weak. **For 89% of PHL proteins, the single closest training protein has the wrong K-type.** For 56% of PHL proteins, no K-type-matching neighbour exists within the top 50.

## What this means for the PHL ceiling

The chain is:
1. ESM-2 mean-pool places PHL proteins alongside training proteins — distance ≈ 0.998 to nearest.
2. But that nearest training protein usually has a different K-type label.
3. The downstream MLP therefore has to classify K-type from embeddings that do not strongly separate K-types.
4. ~0.15 PHL rh@1 is the ceiling of what you can extract from a representation where same-region proteins carry different labels.

**More / bigger embeddings of the same family (ESM-2 15B, ProtT5-XXL) are likely to give more of the same.** The embedding is *information-rich* (functionally well-separated) but *label-orthogonal* (K-types don't have their own clusters).

## Best places to intervene

Ordered by expected impact, given this new understanding:

1. **Segmented pooling preserves local features.** Mean-pooling averages away the discriminative signal; K-type is probably encoded in the receptor-binding domain's local residues. Test: run `phl_neighbor_labels.py` with `esm2_650m_seg4` (and seg8/seg16 once those arrive) and see if top-1 K-type match rate jumps above 10.8%. Seg4 did move PHL rh@1 slightly (0.107 → 0.144 under posList+cl70) — consistent with the local-features hypothesis.
2. **Architecture change — Light Attention / MIL / contrastive.** If K-types cluster by local motifs rather than global mean, a model that attends to specific residues or segments can recover the signal.
3. **Training label audit.** 56% of PHL proteins have no K-matching neighbour in top-50 — some of this is legitimate novelty, some is likely inconsistent K-type annotations across the training set. Spot-check the top-50 neighbours for 10 failing PHL proteins: do the neighbours look like mis-labelled variants of the same functional group?

## What NOT to do

- Extract ESM-2 15B or ProtT5-XXL **as a PHL-improvement move.** Same mean-pool, same problem. (Still worth having for comprehensive benchmarking, but don't expect PHL to move.)
- Keep iterating on filter/sampling/concat of the same mean-pooled embedding. The PHL ceiling is a representation property, not a training-data composition property.

## Artifacts
- `scripts/analysis/phl_training_distance.py` — embedding-space distance analysis.
- `scripts/analysis/phl_neighbor_labels.py` — K-type label agreement at ranks 1/5/10/50.
- `results/analysis/phl_training_distance.{csv,svg}` — raw per-protein similarities.
- `results/analysis/phl_neighbor_labels.{csv,svg}` — per-protein label match + cumulative recall plot.
