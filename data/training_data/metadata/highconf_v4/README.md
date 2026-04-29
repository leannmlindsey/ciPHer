# Highconf v4 — content-cleanup pass on v3_UAT (advisor-curated inclusions + exclusions)

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
| `v4_build_manifest.json` | sha256s of every input + output, full set arithmetic, build timestamp |

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
3. Pair the v4 K list with an O-axis list — for the A/B comparison
   leann is running, the natural O choice is the existing v3
   `HC_O_UAT_multitop.list`, which is unchanged by v4 (the inclusion
   levers don't carry O-labels; A23 exclusions don't impact O). v4
   is currently a **K-side-only** content edit.

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
