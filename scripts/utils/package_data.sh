#!/usr/bin/env bash
# Package raw training and validation inputs into two zip files for
# transfer to Delta-AI and eventual Zenodo upload.
#
# Output: dist/cipher_training_data.zip, dist/cipher_validation_data.zip
# Each zip contains a MANIFEST.txt with sha256 sums so the Delta-AI copy
# can be verified against the laptop copy.
#
# Embeddings are NOT included — regenerate from scripts/ on the target.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DIST="$REPO/dist"
mkdir -p "$DIST"
cd "$REPO"

TRAIN_FILES=(
  data/README.md
  data/COLUMN_STANDARDS.md
  data/training_data/metadata/README.md
  data/training_data/metadata/candidates.faa
  data/training_data/metadata/pipeline_positive.list
  data/training_data/metadata/20250106.K-O.tsv
  data/training_data/metadata/phage_protein_positives.tsv
  data/training_data/metadata/glycan_binders_custom.tsv
  data/training_data/metadata/host_phage_protein_map.tsv
  data/training_data/metadata/candidates_clusters.tsv
  data/training_data/metadata/highconf_tsp_K.list
  data/training_data/metadata/highconf_pipeline_positive_K.list
)

VAL_FILES=(
  data/README.md
  data/COLUMN_STANDARDS.md
  data/validation_data/metadata/README.md
  data/validation_data/metadata/validation_rbps_all.faa
  data/validation_data/metadata/serotypes_merged.tsv
  data/validation_data/HOST_RANGE/STATUS.md
  data/validation_data/HOST_RANGE/MAPPING.md
)
# KlebPhaCol excluded — does not use capsular proteins.
for ds in CHEN GORODNICHIV PBIP PhageHostLearn UCSD; do
  for sub in metadata source; do
    while IFS= read -r f; do
      VAL_FILES+=("$f")
    done < <(find "data/validation_data/HOST_RANGE/$ds/$sub" -type f 2>/dev/null | sort)
  done
done

build_zip() {
  local zip_name="$1"; shift
  local note="$1"; shift
  local files=("$@")

  local zip_path="$DIST/$zip_name"
  local manifest="$REPO/MANIFEST.txt"

  {
    echo "# $zip_name"
    echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# $note"
    echo "# Verify on Delta-AI:  cd <extract_dir> && shasum -a 256 -c MANIFEST.txt"
    echo
    for f in "${files[@]}"; do
      [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 1; }
      shasum -a 256 "$f"
    done
  } > "$manifest"

  rm -f "$zip_path"
  zip -q "$zip_path" MANIFEST.txt "${files[@]}"
  rm "$manifest"

  echo "Built: $zip_path ($(du -h "$zip_path" | cut -f1), ${#files[@]} files)"
}

build_zip "cipher_training_data.zip" \
  "Training inputs (raw only — no embeddings)." \
  "${TRAIN_FILES[@]}"

build_zip "cipher_validation_data.zip" \
  "Validation inputs (5 datasets, no KlebPhaCol, no embeddings)." \
  "${VAL_FILES[@]}"
