# Highconf v4 — content-cleanup pass on v3_UAT (advisor-curated inclusions + exclusions)

> **Per-head canonical files** (use these for cipher-train per-head runs):
> - **`HC_K_v4.list`** (38,162 IDs) — K-head training pool
> - **`HC_O_v4.list`** (37,001 IDs) — O-head training pool
>
> The single-list `pipeline_positive_v4.list` is retained for v1-style
> runs that combine heads. See "Per-head split" section below.



Three protein-ID list variants for **per-head independent training**,
derived by applying advisor-curated inclusion and exclusion levers to
the v3_UAT base. v4 is the first version where the goal is **content
quality** rather than coverage breadth — same approximate size as
v3_UAT (~37 k), different membership.

## Why v4 exists

Three findings that landed during the v3_UAT operating period
identified candidate **bad members in** and **good members out** of
the training pool:

- **A23** (`2026-04-27_md5_klabel_disagreement_passenger_proteins.md`):
  924 cipher proteins carry ≥ 4 distinct K-labels at 100 % sequence
  identity, with one example covering 87 K-types across 21,039
  phages. These are biologically incoherent passengers that
  contribute label noise during training.
- **A28** (`2026-04-28_v4_inclusion_tropiseq_unique_training_proteins.md`):
  1,057 TropiSEQ-trained RBPs are absent from cipher's `candidates.faa`
  even at exact-substring granularity. 90 % carry a single K-label
  (`clean_specific`); the rest carry 2-3 (`clean_polyspecific`).
- **A25** (`2026-04-27_phold_NOT_flagged_phl_homologs.md`): 10 phold
  proteins are near-twins of 32 PHL RBPs but were never flagged by
  any of cipher's 8 RBP-detection tools. They're tool false-negatives
  worth recovering.

v4 applies all three levers as set operations on top of v3_UAT.

## The three list variants

| File | Proteins | K-types | Composition rule |
|---|---:|---:|---|
| `pipeline_positive_v4.list` | **38,162** | 161 / 161 | (HC_K_UAT_multitop − A23) ∪ A25 ∪ A28 — **primary deliverable** |
| `pipeline_positive_v4_no_a23.list` | 38,225 | 161 / 161 | HC_K_UAT_multitop ∪ A25 ∪ A28 (inclusions only — fallback if advisor declines A23 exclusions) |
| `pipeline_positive_v4_clean_only.list` | 37,095 | 161 / 161 | HC_K_UAT_multitop − A23 (exclusions only — useful for isolating the A23 effect) |

## Set arithmetic (audited counts)

| operation | size |
|---|---:|
| HC_K_UAT_multitop (v3 production base) | 37,158 |
| A23 confirmed exclusions (PP-only list) | 924 |
| **A23 ∩ HC_K_UAT_multitop** | **63** |
| A28 TropiSEQ-eligible inclusions | 1,057 |
| A25 phold-NOT_flagged inclusions | 10 |
| A25 ∩ HC_K_UAT_multitop | 0 (disjoint, expected — A25 is 8-tool false-neg) |
| A28 ∩ HC_K_UAT_multitop | 0 (disjoint, expected — A28 is unique-to-TropiSEQ) |

**Note**: only **63** of the 924 A23 candidates appear in v3_UAT —
most passenger proteins fail v3's 95 %-cluster purity gate and are
already excluded. So the A23 lever subtracts 63 IDs from v3_UAT,
not 924. The other 861 A23 candidates live in the broader
pipeline_positive pool but were never v3 training data; they are
flagged for downstream curation if cipher ever revisits looser
training-set recipes.

## Companion files

| File | Purpose |
|---|---|
| `candidates_v4_additions.faa` | 1,067 new RBP sequences (1,057 A28 + 10 A25). The cipher training pipeline should concatenate this onto `data/training_data/metadata/candidates.faa` so v4 list IDs resolve. |
| `host_phage_protein_map_v4.tsv` | Existing 516,839 rows of `host_phage_protein_map.tsv` plus 1,152 new rows (multiple K-locus atoms expanded to multiple rows per protein). All new K-labels normalised `KL → K`. |
| `candidates_clusters_v4.tsv` | Existing 143,240 rows of `candidates_clusters.tsv` verbatim + 1,067 new rows with cluster IDs at 9 identity thresholds (cl30/40/50/60/70/80/85/90/95). New IDs **inherit** an existing cipher cluster ID where the new protein has a ≥ T % identity match in cipher's pool, otherwise are **fresh** (allocated starting at `max_existing_at_T + 1`). Cipher's training pipeline reads this for cluster-stratified round-robin downsampling at `cluster_threshold=70`. |
| `candidates_clusters_v4.summary.json` | Inheritance counts per threshold (e.g. at cl70: 454 inherited / 221 fresh of 1,067) |
| `v4_build_manifest.json` | sha256s of every input + output, full set arithmetic, source-dataset provenance, cluster-augmentation method, build timestamp |

## K-locus normalisation

TropiSEQ uses the `KL` prefix; cipher's existing
`host_phage_protein_map.tsv` mixes `K` (364,509 rows) and `KL`
(130,831 rows). Per the v4 build spec, new TropiSEQ records are
normalised **`KL{n} → K{n}`** so they land under the dominant prefix
and are not silently dropped as "novel K-types" by cipher's
`prepare_training_data` step.

A25 records carry K-prefix labels in their FASTA headers already (the
phold-target K-types are derived from PHL phage assignments which are
stored K-prefix); no normalisation needed.

## How to use the v4 list

1. Concatenate `candidates_v4_additions.faa` onto the existing
   `candidates.faa` so the 1,067 inclusion IDs resolve to sequences:
   ```
   cat data/training_data/metadata/candidates.faa \
       data/training_data/metadata/highconf_v4/candidates_v4_additions.faa \
     > data/training_data/metadata/candidates_v4.faa
   ```
2. Use `host_phage_protein_map_v4.tsv` as the label map (the existing
   schema, just with new rows appended).
3. Use `candidates_clusters_v4.tsv` as the cluster file (drop-in
   replacement for `candidates_clusters.tsv`). Cipher's training
   pipeline reads it at `cluster_threshold=70` for cluster-stratified
   round-robin downsampling under `max_samples_per_k=1000`. Without
   the v4 cluster file, the new proteins would be treated as
   singletons and silently over-weighted in round-robin sampling.
4. Pair the v4 K list with an O-axis list — for the A/B comparison
   leann is running, the natural O choice is the existing v3
   `HC_O_UAT_multitop.list`, which is unchanged by v4 (the inclusion
   levers don't carry O-labels; A23 exclusions don't impact O). v4
   is currently a **K-side-only** content edit.

## Per-head split (HC_K_v4 / HC_O_v4) — added 2026-05-03

Per agent 1's 2026-05-03 ask, v4 follows the v2/v3 per-head training
convention. Cipher-train uses two lists, one per head, with per-head
loss masking. Motivation: finding
`2026-04-22_k_and_o_heads_independent_on_phl.md` (4b) — losing either
head costs ~30 % of PHL HR@1.

| File | Size | Recipe |
|---|---:|---|
| `HC_K_v4.list` | **38,162** | `(HC_K_UAT_multitop − A23) ∪ A25 ∪ A28`  (= identical to `pipeline_positive_v4.list`; renamed for the per-head convention) |
| `HC_O_v4.list` | **37,001** | `HC_O_UAT_multitop − (A23 ∩ HC_O_UAT_multitop)`  (177 passengers subtracted; no K-axis inclusions on O — A25/A28 lack clean O-labels) |

**Why no inclusions on the O side:** A28 (TropiSEQ-unique) and A25
(phold-NOT_flagged) candidates were extracted under K-axis logic and
do not carry biologically-clean O-axis labels. Adding them to the
O-head training pool would inject noise. The O-head benefits from
v4 only via the A23 cleanup (177 passenger proteins removed from the
v3 O pool).

**Cipher-train invocation:**
```
--positive_list_k data/training_data/metadata/highconf_v4/HC_K_v4.list
--positive_list_o data/training_data/metadata/highconf_v4/HC_O_v4.list
```

The single-list `pipeline_positive_v4.list` is retained for v1-style
runs that combine heads on one list.

## Cluster augmentation method (relevant for v5 and beyond)

Cluster IDs for the 1,067 new proteins were produced by the
augmentation script at
`CLAUDE_PHI_DATA_ANALYSIS/analyses/30_dataset_version_history/work/v4_build/augment_clusters.py`.
The script is **generic** — same interface as v4, drop in any new
FASTA + reference cluster file → produces an augmented cluster file.
Algorithm at each threshold T:

1. mmseqs2 `easy-search` of new sequences against cipher's
   `candidates.faa` with `--min-seq-id T --cov-mode 2 -c 0.8`
   (CD-HIT-style coverage of shorter sequence).
2. For each new protein, take the best hit (highest bitscore; pident
   tiebreak). If best hit pident ≥ T, **inherit** that target's
   cluster ID at threshold T from the existing cluster file.
3. For new proteins with no hit ≥ T, mmseqs2 `easy-cluster` on the
   no-hit subset at T → **fresh** cluster IDs allocated starting at
   `max(existing_at_T) + 1`.

Inheritance summary across 1,067 new proteins:

| threshold | inherited | fresh groups |
|---:|---:|---:|
| 30 % | 582 | 130 |
| 40 % | 513 | 155 |
| 50 % | 475 | 195 |
| 60 % | 462 | 214 |
| **70 %** | **454** | **221** |
| 80 % | 453 | 225 |
| 85 % | 451 | 231 |
| 90 % | 448 | 241 |
| 95 % | 426 | 278 |

At cl70 (cipher's stratification threshold), 454 of 1,067 new
proteins (43 %) have ≥ 70 % identity neighbours in cipher's existing
pool and inherit those clusters; the remaining 613 form 221 fresh
clusters (mean ~2.8 members per fresh cluster). The pattern is
expected: A28 candidates were specifically the TropiSEQ-unique set,
but 43 % still have a ≥ 70 % identity cousin somewhere in cipher's
broader 143k-pool — those are RBPs whose phages cipher has via a
different source genome. The 57 % fresh-cluster fraction is the
genuinely-novel-to-cipher contribution.

## Source datasets for the 1,067 inclusions

(provenance recorded in `v4_build_manifest.json`)

- **A28 source** (1,057 proteins): TropiSEQ training table at
  `/Users/leannmlindsey/WORK/PHI_TSP/DpoTropiSearch_zenoto_data/TropiGATv2.final_df_v2.tsv`
  (21,350 rows → 4,084 unique full proteins; 1,057 are eligible
  TropiSEQ-unique candidates).
- **A25 source** (10 proteins): cipher's phold predictions at
  `/Users/leannmlindsey/WORK/PHI_TSP/phi_tsp/klebsiella/data/proteins/phold.rmdup.faa`
  (~1.36M sequences — the corpus larger than `pipeline_positive`'s
  69k that contains tool-FN candidates not in the 8-tool union).

## A/B against v3_UAT

The headline experiment leann's running:
- v3_UAT: HC_K_UAT_multitop + HC_O_UAT_multitop (current production)
- v4: pipeline_positive_v4 + HC_O_UAT_multitop (this directory)

Same architecture (light_attention + ProtT5-xl-seg8), same hyper-
parameters. Expected lift on PHL is concentrated in K-types where
TropiSEQ added training coverage (KL64 / KL107 / KL2 dominate the
A28 inclusion pool). PHL has 10 K-types TropiSEQ doesn't cover —
those don't benefit from v4. Expected magnitude: small but
identifiable above retrieval baseline noise (per A25 finding's
"v3_UAT is the first recipe to beat retrieval on PHL" framing,
v4 should sit a few pp higher).

## Provenance

- A23 (exclusion lever):
  `CLAUDE_PHI_DATA_ANALYSIS/analyses/23_md5_klabel_disagreement/`
- A25 (phold inclusion lever):
  `CLAUDE_PHI_DATA_ANALYSIS/analyses/25_phl_phold_novel_candidates/`
- A28 (TropiSEQ inclusion lever):
  `CLAUDE_PHI_DATA_ANALYSIS/analyses/28_v4_candidate_additions/`
- v4 build script + manifest:
  `CLAUDE_PHI_DATA_ANALYSIS/analyses/30_dataset_version_history/work/v4_build/build_v4.py`
- v3_UAT base:
  `data/training_data/metadata/highconf_v3_multitop/HC_K_UAT_multitop.list`

## Pending advisor sign-off

The A23 exclusion lever is the only piece of v4 that requires advisor
review (the 924 → 63-in-HC subtraction). The scatter for tomorrow's
PI meeting at `cipher/results/figures/passenger_proteins_scatter.png`
is the evidence package. If the advisor declines A23, the
`pipeline_positive_v4_no_a23.list` variant is the fallback — same
inclusions, no exclusions.

A28 and A25 inclusions are already advisor-greenlit at the candidate-
package level (per the A28 / A25 advisor-review materials). The list
build doesn't add anything beyond what those candidate packages
already proposed.
