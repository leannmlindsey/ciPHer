import os
from pathlib import Path

REPO_PATH = Path(__file__).parent.parent.resolve()

ESM_PATH = REPO_PATH.parent.resolve() / "models/esm/"
MODEL_PATH = REPO_PATH.parent.resolve() / "models/fine-tuned/"

ESM_MODELS = {
	't6':  'facebook/esm2_t6_8M_UR50D',
	't12': 'facebook/esm2_t12_35M_UR50D',
	't30': 'facebook/esm2_t30_150M_UR50D',
	't33': 'facebook/esm2_t33_650M_UR50D',
	't36': 'facebook/esm2_t36_3B_UR50D',
	't48': 'facebook/esm2_t48_15B_UR50D',
}

PHROG_CLASSES = [
	"DNA, RNA and nucleotide metabolism",
	"tail",
	"head and packaging",
	"transcription regulation",
	"connector",
	"lysis",
	"moron, auxiliary metabolic gene and host takeover",
	"integration and excision",
	"other",
]
CLASS_IDS = {c: i for i, c in enumerate(PHROG_CLASSES)}