#!/usr/bin/env bash
#
# Re-run per_head_strict_eval on the BROADER PHL set (~8,223 proteins
# vs the strict 8/8 tail of 256) for the 3 models in agent 4's H5 spec:
#   1. la_v3_uat_prott5_xl_seg8         — production v3_UAT
#   2. sweep_posList_esm2_3b_mean_cl70  — best K-head from hybrid
#   3. sweep_kmer_aa20_k4               — best single + best O-head
#
# Pre-flight (caller's responsibility — script fails early if missing):
#   - Broader val embedding NPZs at the H5-extraction output paths
#     (run scripts/run_h5_broader_phl_extractions.sh first).
#   - Broader phage_protein_mapping CSV at the path below
#     (run scripts/analysis/build_broader_phl_mapping.py first).
#
# Output JSONs are written WITH a `_broad_phl` suffix INTO the existing
# experiment results/ dirs so they sit next to the strict JSONs without
# overwriting.
#
# Usage:
#   bash scripts/run_h5_broader_phl_eval.sh                # all 3 models
#   MODELS="prott5"          bash scripts/run_h5_broader_phl_eval.sh
#   MODELS="esm2 kmer"       bash scripts/run_h5_broader_phl_eval.sh
#   DRY_RUN=1                bash scripts/run_h5_broader_phl_eval.sh

set -euo pipefail

ACCOUNT="${ACCOUNT:-bfzj-dtai-gh}"
PARTITION="${PARTITION:-ghx4}"
CONDA_ENV="${CONDA_ENV:-esmfold2}"
CIPHER_DIR="${CIPHER_DIR:-/projects/bfzj/llindsey1/PHI_TSP/ciPHer}"

# H5 broader inputs
BROAD_MAPPING="${BROAD_MAPPING:-${CIPHER_DIR}/data/validation_data/HOST_RANGE/PhageHostLearn/metadata/phage_protein_mapping_broad_phl.csv}"
BROAD_FASTA="${BROAD_FASTA:-/projects/bfzj/llindsey1/PHI_TSP/phi_tsp/klebsiella/PI_INFO/phagehostlearn_phold_aa.fasta}"
BROAD_PROTT5="${BROAD_PROTT5:-/work/hdd/bfzj/llindsey1/validation_embeddings/prott5_xl_segments8/validation_embeddings_broad_phl_md5.npz}"
BROAD_ESM2_3B="${BROAD_ESM2_3B:-/work/hdd/bfzj/llindsey1/validation_embeddings_esm2_3b/validation_embeddings_broad_phl_md5.npz}"
BROAD_KMER="${BROAD_KMER:-/work/hdd/bfzj/llindsey1/kmer_features/validation_aa20_k4_broad_phl.npz}"

# Strict inputs (existing 5-dataset val tree)
STRICT_VAL_DS="${CIPHER_DIR}/data/validation_data/HOST_RANGE"

# Light Attention experiments live in a sibling worktree, not in
# cipher/experiments/attention_mlp/.
LA_WORKTREE="${LA_WORKTREE:-/projects/bfzj/llindsey1/PHI_TSP/cipher-light-attention}"

LOG_DIR="${CIPHER_DIR}/scripts/_logs/h5_eval"
mkdir -p "$LOG_DIR"

MODELS="${MODELS:-prott5 esm2 kmer}"
DRY_RUN="${DRY_RUN:-0}"

# Pre-flight check on broader inputs
if [[ ! -f "$BROAD_MAPPING" ]]; then
    echo "ERROR: broader mapping CSV not found at $BROAD_MAPPING" >&2
    echo "  build it first: python scripts/analysis/build_broader_phl_mapping.py ..." >&2
    exit 1
fi

submit_eval() {
    local MODEL_KEY="$1"
    local EXP_NAME EMB_FILE EXP_DIR
    case "$MODEL_KEY" in
        prott5)
            EXP_NAME="la_v3_uat_prott5_xl_seg8"
            EMB_FILE="$BROAD_PROTT5"
            EXP_DIR="${LA_WORKTREE}/experiments/light_attention/${EXP_NAME}"
            ;;
        esm2)
            EXP_NAME="sweep_posList_esm2_3b_mean_cl70"
            EMB_FILE="$BROAD_ESM2_3B"
            EXP_DIR="${CIPHER_DIR}/experiments/attention_mlp/${EXP_NAME}"
            ;;
        kmer)
            EXP_NAME="sweep_kmer_aa20_k4"
            EMB_FILE="$BROAD_KMER"
            EXP_DIR="${CIPHER_DIR}/experiments/attention_mlp/${EXP_NAME}"
            ;;
        *)
            echo "WARNING: unknown MODELS key '$MODEL_KEY' (expected: prott5|esm2|kmer)" >&2
            return
            ;;
    esac

    if [[ ! -d "$EXP_DIR" ]]; then
        echo "WARNING: experiment dir missing: $EXP_DIR — skipping $MODEL_KEY" >&2
        return
    fi
    if [[ ! -f "$EMB_FILE" ]]; then
        echo "ERROR: broader embedding NPZ missing: $EMB_FILE" >&2
        echo "  run scripts/run_h5_broader_phl_extractions.sh first" >&2
        return 1
    fi

    local JOB_NAME="h5_eval_${MODEL_KEY}"
    local OUT_JSON="${EXP_DIR}/results/per_head_strict_eval_broad_phl.json"
    local PER_PHAGE="${CIPHER_DIR}/results/analysis/per_phage/per_phage_${EXP_NAME}_broad_phl.tsv"
    mkdir -p "$(dirname "$PER_PHAGE")"

    # Build per-job temp val-datasets-dir: symlink the 4 unchanged datasets,
    # build a real PhageHostLearn dir with the broader mapping swapped in.
    # The temp dir lives under $LOG_DIR/temp_val_<job>/ so concurrent jobs
    # don't clobber each other.
    local CMD="
        source \$(conda info --base)/etc/profile.d/conda.sh
        conda activate ${CONDA_ENV}
        cd ${CIPHER_DIR}
        export PYTHONPATH=${CIPHER_DIR}/src:\${PYTHONPATH:-}

        TEMP_VAL_DS=\$(mktemp -d -p ${LOG_DIR} val_${MODEL_KEY}.XXXXXX)
        echo \"  temp val ds dir: \$TEMP_VAL_DS\"

        for ds in CHEN GORODNICHIV UCSD PBIP; do
            ln -s ${STRICT_VAL_DS}/\$ds \$TEMP_VAL_DS/\$ds
        done

        # Build broader PhageHostLearn dir: symlink everything from strict
        # dir, then overlay the broader mapping CSV in metadata/.
        BROAD_PHL=\$TEMP_VAL_DS/PhageHostLearn
        mkdir -p \$BROAD_PHL/metadata
        for f in ${STRICT_VAL_DS}/PhageHostLearn/*; do
            base=\$(basename \$f)
            if [ \"\$base\" = \"metadata\" ]; then
                continue
            fi
            ln -s \$f \$BROAD_PHL/\$base
        done
        for f in ${STRICT_VAL_DS}/PhageHostLearn/metadata/*; do
            base=\$(basename \$f)
            if [ \"\$base\" = \"phage_protein_mapping.csv\" ]; then
                continue
            fi
            ln -s \$f \$BROAD_PHL/metadata/\$base
        done
        cp ${BROAD_MAPPING} \$BROAD_PHL/metadata/phage_protein_mapping.csv

        python ${CIPHER_DIR}/scripts/analysis/per_head_strict_eval.py ${EXP_DIR} \
            --datasets PhageHostLearn \
            --val-fasta ${BROAD_FASTA} \
            --val-embedding-file ${EMB_FILE} \
            --val-datasets-dir \$TEMP_VAL_DS \
            --out-json ${OUT_JSON} \
            --per-phage-out ${PER_PHAGE}

        rm -rf \$TEMP_VAL_DS
    "

    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY [$JOB_NAME exp=$EXP_NAME]:"
        echo "  emb:   $EMB_FILE"
        echo "  out:   $OUT_JSON"
        echo "  perph: $PER_PHAGE"
        return
    fi

    sbatch \
        --account="$ACCOUNT" --partition="$PARTITION" \
        --gpus-per-node=1 --cpus-per-task=4 --mem=48G --time=04:00:00 \
        --job-name="$JOB_NAME" \
        --output="${LOG_DIR}/${JOB_NAME}_%j.log" \
        --wrap="$CMD"
}

for m in $MODELS; do
    submit_eval "$m"
done
