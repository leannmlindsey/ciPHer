#!/usr/bin/env bash
#
# Submit embedding-extraction SLURM jobs on Delta-AI.
#
# Edit the EXTRACTIONS array to add new (model, pooling) combos. Each row
# fans out TWO jobs: one for the training FASTA and one for the validation
# FASTA, with matching output paths.
#
# Usage:
#   # Submit everything in EXTRACTIONS:
#   bash scripts/extract_embeddings/submit_extractions.sh
#
#   # Single combo by its derived label (<model_tag>_<pooling>):
#   bash scripts/extract_embeddings/submit_extractions.sh esm2_650m_segments8
#
#   # Dry run:
#   DRY_RUN=1 bash scripts/extract_embeddings/submit_extractions.sh

set -euo pipefail

# ============================================================
# Delta-AI configuration
# ============================================================
ACCOUNT="bfzj-dtai-gh"
PARTITION="ghx4"
# Per-family conda envs (override with family-specific env vars).
# esmfold2 has torch + fair-esm; prott5 has torch + transformers.
ESM2_ENV="${ESM2_ENV:-esmfold2}"
PROTT5_ENV="${PROTT5_ENV:-prott5}"
CIPHER_DIR="/projects/bfzj/llindsey1/PHI_TSP/ciPHer"

family_env() {
    case "$1" in
        esm2)   echo "$ESM2_ENV" ;;
        prott5) echo "$PROTT5_ENV" ;;
        *)      echo "ERROR: unknown family '$1'" >&2; exit 1 ;;
    esac
}

TRAIN_FASTA="${CIPHER_DIR}/data/training_data/metadata/candidates.faa"
VAL_FASTA="${CIPHER_DIR}/data/validation_data/metadata/validation_rbps_all.faa"

# Output roots (work/ for fast scratch, not /projects/)
EMB_ROOT="/work/hdd/bfzj/llindsey1/embeddings"
VAL_EMB_ROOT="/work/hdd/bfzj/llindsey1/validation_embeddings"

# ============================================================
# Extractions to submit.
#
# Format (space-separated columns):
#   "family  model_name  pooling  mem  gpus  time  extra_args"
#
# family: esm2 or prott5 (dispatches to the right extract_*.py)
# model_name: HuggingFace model id (esm2_t33_650M_UR50D,
#             Rostlab/prot_t5_xl_uniref50, etc.)
# pooling: mean | segmentsN (any N>=2)
# mem: SLURM --mem value; use "0" for full-node reservation
# gpus: --gpus-per-node (1 normally, 4 for mem=0)
# time: SLURM --time limit
# extra_args: extra CLI args for the extract script (use "-" for none).
#             E.g. "--half_precision" for very large models.
# ============================================================
EXTRACTIONS=(
    # ESM-2 650M seg8 / seg16 — completed 2026-04-20, kept here for history:
    # "esm2    esm2_t33_650M_UR50D            segments8   64G   1  12:00:00  -"
    # "esm2    esm2_t33_650M_UR50D            segments16  64G   1  12:00:00  -"

    # ESM-2 650M per-residue (full) — needed for light_attention_binary and
    # the light-attention models, which pool attention across all residues.
    # The existing file at /work/hdd/.../embeddings_full/ covered only 4453
    # MD5s (see 2026-04-21 training run: model trained on 14% of the filtered
    # set). mem=0 because the extraction holds all (L, 1280) arrays in a dict
    # before np.savez_compressed — ~60-80 GB RAM for the full candidates set.
    "esm2    esm2_t33_650M_UR50D            full        0     1  24:00:00  -"

    # ProtT5-XL segmented pooling (tests whether ProtT5's K-type signal
    # amplifies with local pooling — ESM-2 seg4 bumped top-1 match 11.3 -> 12.6).
    # --half_precision halves model VRAM; --max_length 3000 caps the attention
    # matrix for a small number of very-long sequences (>4000 aa) that OOM'd
    # the H100 on the first attempt near the tail of the length-sorted FASTA.
    "prott5  Rostlab/prot_t5_xl_uniref50    segments4   128G  1  12:00:00  --half_precision --max_length 3000"
    "prott5  Rostlab/prot_t5_xl_uniref50    segments8   128G  1  12:00:00  --half_precision --max_length 3000"
    "prott5  Rostlab/prot_t5_xl_uniref50    segments16  128G  1  12:00:00  --half_precision --max_length 3000"

    # ProtT5-XXL mean (11B params — does size help for ProtT5?).
    # Half precision required to fit on a single H100. max_length guard too.
    "prott5  Rostlab/prot_t5_xxl_uniref50   mean        0     4  24:00:00  --half_precision --max_length 3000"
)

FILTER="${1:-}"
DRY_RUN="${DRY_RUN:-0}"

# Map HF model name -> short tag for output paths.
model_tag() {
    case "$1" in
        esm2_t30_150M_UR50D)          echo "esm2_150m" ;;
        esm2_t33_650M_UR50D)          echo "esm2_650m" ;;
        esm2_t36_3B_UR50D)            echo "esm2_3b" ;;
        esm2_t48_15B_UR50D)           echo "esm2_15b" ;;
        Rostlab/prot_t5_xl_uniref50)  echo "prott5_xl" ;;
        Rostlab/prot_t5_xl_bfd)       echo "prott5_xl_bfd" ;;
        Rostlab/prot_t5_xxl_uniref50) echo "prott5_xxl" ;;
        *) echo "$1" | tr '/' '_' ;;
    esac
}

# Map ESM-2 model -> final-layer index (layer that the sweep uses).
esm2_layer() {
    case "$1" in
        esm2_t30_150M_UR50D) echo 30 ;;
        esm2_t33_650M_UR50D) echo 33 ;;
        esm2_t36_3B_UR50D)   echo 36 ;;
        esm2_t48_15B_UR50D)  echo 48 ;;
        *) echo "ERROR: unknown ESM-2 layer for model $1" >&2; exit 1 ;;
    esac
}

build_and_submit() {
    local fasta="$1"; shift
    local output_npz="$1"; shift
    local name="$1"; shift
    local family="$1"; shift
    local model="$1"; shift
    local pooling="$1"; shift
    local mem="$1"; shift
    local gpus="$1"; shift
    local time="$1"; shift
    local extra_py_args="$1"; shift

    local extract_py=""
    local model_arg=""
    local pooling_arg=""
    case "$family" in
        esm2)
            extract_py="${CIPHER_DIR}/scripts/extract_embeddings/esm2_extract.py"
            local layer=$(esm2_layer "$model")
            model_arg="--model ${model} --layer ${layer}"
            pooling_arg="--pooling ${pooling}"
            ;;
        prott5)
            extract_py="${CIPHER_DIR}/scripts/extract_embeddings/prott5_extract.py"
            model_arg="--model_name ${model}"
            pooling_arg="--pooling ${pooling}"
            ;;
        *)
            echo "ERROR: unknown family '$family'" >&2
            exit 1
            ;;
    esac

    local conda_env=$(family_env "$family")
    local cmd="python ${extract_py} ${fasta} ${output_npz} ${model_arg} ${pooling_arg} --key_by_md5 ${extra_py_args}"

    local job_script="#!/bin/bash
#SBATCH --job-name=${name}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=${gpus}
#SBATCH --cpus-per-task=8
#SBATCH --mem=${mem}
#SBATCH --time=${time}
#SBATCH --output=${CIPHER_DIR}/logs/${name}_%j.log
#SBATCH --error=${CIPHER_DIR}/logs/${name}_%j.log

set -euo pipefail

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate ${conda_env}
cd ${CIPHER_DIR}
export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}
export TORCH_HOME=/projects/bfzj/llindsey1/RBP_Structural_Similarity/models
export TRANSFORMERS_CACHE=\${TORCH_HOME}

echo \"======================================\"
echo \"Extract: ${name}\"
echo \"  Input:  ${fasta}\"
echo \"  Output: ${output_npz}\"
echo \"  Model:  ${model}  pooling=${pooling}\"
echo \"  Started: \$(date)\"
echo \"======================================\"

mkdir -p \$(dirname ${output_npz})

${cmd}

echo \"======================================\"
echo \"Done: ${name} at \$(date)\"
echo \"======================================\"
"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] ${name}  (mem=${mem}, gpus=${gpus}, time=${time})"
        echo "    cmd: ${cmd}"
        echo ""
    else
        mkdir -p "${CIPHER_DIR}/logs"
        local job_id=$(echo "$job_script" | sbatch | awk '{print $NF}')
        echo "  Submitted ${job_id} — ${name}"
    fi
}

echo "============================================================"
echo "EMBEDDING EXTRACTION"
echo "  Cipher:  ${CIPHER_DIR}"
echo "  Train FASTA:      ${TRAIN_FASTA}"
echo "  Validation FASTA: ${VAL_FASTA}"
echo "============================================================"
echo ""

N=0
for entry in "${EXTRACTIONS[@]}"; do
    read -r FAMILY MODEL POOLING MEM GPUS TIME EXTRA_ARGS <<< "$entry"
    # Dash means "no extra args" — don't pass literal "-" to python.
    [ "$EXTRA_ARGS" = "-" ] && EXTRA_ARGS=""
    TAG=$(model_tag "$MODEL")
    LABEL="${TAG}_${POOLING}"

    if [ -n "$FILTER" ] && [ "$LABEL" != "$FILTER" ]; then
        continue
    fi

    TRAIN_DIR="${EMB_ROOT}/${TAG}_${POOLING}"
    VAL_DIR="${VAL_EMB_ROOT}/${TAG}_${POOLING}"
    TRAIN_NPZ="${TRAIN_DIR}/candidates_${LABEL}_md5.npz"
    VAL_NPZ="${VAL_DIR}/validation_${LABEL}_md5.npz"

    build_and_submit "$TRAIN_FASTA" "$TRAIN_NPZ" "extract_${LABEL}_train" \
        "$FAMILY" "$MODEL" "$POOLING" "$MEM" "$GPUS" "$TIME" "$EXTRA_ARGS"
    build_and_submit "$VAL_FASTA" "$VAL_NPZ" "extract_${LABEL}_val" \
        "$FAMILY" "$MODEL" "$POOLING" "$MEM" "$GPUS" "$TIME" "$EXTRA_ARGS"
    N=$((N + 2))
done

echo ""
if [ "$DRY_RUN" = "1" ]; then
    echo "DRY RUN complete. Set DRY_RUN=0 to submit."
else
    echo "Submitted ${N} extraction jobs. Monitor: squeue -u \$USER"
fi
