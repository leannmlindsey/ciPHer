import argparse
import hashlib
import torch
import pandas as pd
import numpy as np
from Bio import SeqIO
import sys
from pathlib import Path
from tqdm import tqdm
import os
# Model configurations with their max sequence lengths
ESM_MODEL_MAX_LENGTHS = {
    # ESM-2 models
    "esm2_t6_8M_UR50D": 1024,
    "esm2_t12_35M_UR50D": 1024,
    "esm2_t30_150M_UR50D": 1024,
    "esm2_t33_650M_UR50D": 1024,  # Can handle more with sufficient memory
    "esm2_t36_3B_UR50D": 1024,    # Can be extended to 2048
    "esm2_t48_15B_UR50D": 1024,   # Can be extended to 2048
    # ESM-1 models (if needed)
    "esm1_t6_43M_UR50S": 1024,
    "esm1_t12_85M_UR50S": 1024,
    "esm1_t34_670M_UR50S": 1024,
    "esm1b_t33_650M_UR50S": 1024,
    # ESM-1v models
    "esm1v_t33_650M_UR90S_1": 1024,
    # ESM-C models (optimized for sequence generation)
    "esmc_300m": 1024,
    "esmc_600m": 1024,
    "esmc_6b": 2048,  # 6B model supports longer sequences
    "esmc_300m_202412": 1024,  # December 2024 release
    "esmc_600m_202412": 1024,  # December 2024 release
    "esmc_6b_202412": 2048,  # December 2024 release
}

# Extended limits for larger models with sufficient memory
ESM_MODEL_EXTENDED_LENGTHS = {
    "esm2_t33_650M_UR50D": 2048,
    "esm2_t36_3B_UR50D": 2048,
    "esm2_t48_15B_UR50D": 2048,
    # ESM-C models can potentially handle longer sequences
    "esmc_600m": 2048,
    "esmc_600m_202412": 2048,
    "esmc_6b": 4096,  # 6B model may handle even longer with sufficient memory
    "esmc_6b_202412": 4096,
}

# Model type detection patterns
ESM_MODEL_TYPES = {
    "esm2": ["esm2_t"],
    "esm1": ["esm1_t", "esm1b_t"],
    "esm1v": ["esm1v_t"],
    "esmc": ["esmc_"]
}

def detect_model_type(model_name):
    """
    Detect the type of ESM model from its name.

    Args:
        model_name: Name of the ESM model

    Returns:
        Model type string ('esm2', 'esm1', 'esm1v', 'esmc')
    """
    model_name_lower = model_name.lower()
    for model_type, patterns in ESM_MODEL_TYPES.items():
        if any(pattern in model_name_lower for pattern in patterns):
            return model_type
    return "esm2"  # Default to ESM-2

def load_esmc_model(model_name="esmc_600m"):
    """
    Load ESM-C model for sequence generation and embedding extraction.

    Args:
        model_name: Name of the ESM-C model

    Note: ESM-C models require special handling as they're designed for generation
    """
    print(f"Loading ESM-C model: {model_name}")

    # Use TORCH_HOME if already set, otherwise fall back to default
    cache_dir = os.environ.get('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TORCH_HOME'] = cache_dir
    os.environ['HF_HOME'] = os.environ.get('HF_HOME', cache_dir)

    try:
        # Try to import ESM-C specific modules if available
        try:
            from esm.models import esmc
            print("Using ESM-C specific loader")

            # Load ESM-C model
            if "300m" in model_name:
                model = esmc.ESMC_300M()
            elif "600m" in model_name:
                model = esmc.ESMC_600M()
            elif "6b" in model_name:
                model = esmc.ESMC_6B()
            else:
                print(f"Unknown ESM-C model: {model_name}, defaulting to 600M")
                model = esmc.ESMC_600M()

            # ESM-C models use a different alphabet
            alphabet = model.alphabet

        except ImportError:
            print("ESM-C specific modules not found, trying torch.hub")
            # Fallback to torch.hub if available
            model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)

    except Exception as e:
        print(f"Error loading ESM-C model: {e}")
        print("Falling back to ESM-2 model")
        model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        model_name = "esm2_t33_650M_UR50D"

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")

    # ESM-C models may have different layer structure
    if hasattr(model, 'num_layers'):
        print(f"Model has {model.num_layers} layers")
    else:
        print("Model layer structure differs from ESM-2")

    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    return model, alphabet, device, model_name
def load_esm2_model_with_checkpoint(checkpoint_path, model_name="esm2_t33_650M_UR50D"):
    """
    Load ESM-2 model using torch.hub and then load checkpoint weights.

    Args:
        checkpoint_path: Path to checkpoint file
        model_name: Name of the ESM model to load
    """
    print(f"Loading ESM-2 model architecture via torch.hub...")
    print(f"Model: {model_name}")

    # Load model architecture from torch.hub
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)

    print(f"Loading checkpoint weights from: {checkpoint_path}")

    try:
        # Load checkpoint with weights_only=False since we trust ESM checkpoints
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Load the state dict
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("Loaded weights from checkpoint['model']")
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("Loaded weights from checkpoint['state_dict']")
        else:
            # Assume the checkpoint is the state dict itself
            model.load_state_dict(checkpoint)
            print("Loaded weights from checkpoint directly")

    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        print("Trying to load without custom weights (using pretrained weights)...")
        # Just use the pretrained model if checkpoint loading fails
        pass

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")
    print(f"Model has {model.num_layers} layers")
    return model, alphabet, device, model_name

def load_esm2_model_pretrained(model_name="esm2_t48_15B_UR50D"):
    """
    Load ESM model using torch.hub (no checkpoint).
    Automatically handles ESM-2, ESM-1, and ESM-C models.
    Cache to /data directory to avoid home quota limits.

    Args:
        model_name: Name of the ESM model to load
    """
    # Detect model type
    model_type = detect_model_type(model_name)

    if model_type == "esmc":
        # Use ESM-C specific loader
        return load_esmc_model(model_name)

    # For ESM-1 and ESM-2 models
    # Use TORCH_HOME if already set, otherwise fall back to default
    cache_dir = os.environ.get('TORCH_HOME', os.path.join(os.path.expanduser('~'), '.cache', 'torch'))
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TORCH_HOME'] = cache_dir
    os.environ['HF_HOME'] = os.environ.get('HF_HOME', cache_dir)

    print(f"Using cache directory: {cache_dir}")
    print(f"Loading {model_type.upper()} model via torch.hub...")
    print(f"Model: {model_name}")

    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)

    # Use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")
    print(f"Model has {model.num_layers} layers")

    if torch.cuda.is_available():
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    return model, alphabet, device, model_name

def get_max_length_for_model(model_name, use_extended=False):
    """
    Get the maximum sequence length for a given model.

    Args:
        model_name: Name of the ESM model
        use_extended: Whether to use extended length (if available)

    Returns:
        Maximum sequence length
    """
    if use_extended and model_name in ESM_MODEL_EXTENDED_LENGTHS:
        max_length = ESM_MODEL_EXTENDED_LENGTHS[model_name]
        print(f"Using extended max length for {model_name}: {max_length}")
    elif model_name in ESM_MODEL_MAX_LENGTHS:
        max_length = ESM_MODEL_MAX_LENGTHS[model_name]
        print(f"Using standard max length for {model_name}: {max_length}")
    else:
        # Default to 1024 if model not recognized
        max_length = 1024
        print(f"Model {model_name} not recognized, using default max length: {max_length}")

    return max_length

def analyze_sequence_lengths(fasta_file, max_length):
    """
    Analyze sequence lengths in FASTA file and report truncation statistics.

    Args:
        fasta_file: Path to FASTA file
        max_length: Maximum sequence length that will be used

    Returns:
        Dictionary with statistics
    """
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    lengths = [len(str(record.seq)) for record in sequences]

    truncated = [l for l in lengths if l > max_length - 2]  # -2 for special tokens

    stats = {
        "total_sequences": len(sequences),
        "max_length": max(lengths) if lengths else 0,
        "mean_length": np.mean(lengths) if lengths else 0,
        "median_length": np.median(lengths) if lengths else 0,
        "truncated_count": len(truncated),
        "truncated_percentage": (len(truncated) / len(sequences) * 100) if sequences else 0
    }

    if truncated:
        stats["max_residues_lost"] = max(lengths) - (max_length - 2)
        stats["mean_residues_lost"] = np.mean([l - (max_length - 2) for l in truncated])
    else:
        stats["max_residues_lost"] = 0
        stats["mean_residues_lost"] = 0

    return stats

def get_protein_embedding(sequence, model, alphabet, device, layer, max_length=1024, model_type=None, pooling="mean"):
    """
    Generate embedding for a single protein sequence from a specific layer.
    Handles ESM-2, ESM-1, and ESM-C models.

    Args:
        sequence: Protein sequence string
        model: ESM model
        alphabet: ESM alphabet
        device: torch device
        layer: Layer number to extract embeddings from (may be ignored for ESM-C)
        max_length: Maximum sequence length
        model_type: Type of model ('esm2', 'esm1', 'esmc', etc.)
        pooling: Pooling strategy — 'mean' (default, 1D vector) or 'full' (per-residue L×D)

    Returns:
        If pooling='mean': 1D numpy array (embedding_dim,)
        If pooling='full': 2D numpy array (seq_length, embedding_dim)
    """
    # Truncate sequence if too long
    if len(sequence) > max_length - 2:  # -2 for special tokens
        sequence = sequence[:max_length - 2]
        tqdm.write(f"Warning: Sequence truncated to {max_length - 2} residues")

    # Prepare batch converter
    batch_converter = alphabet.get_batch_converter()

    # Prepare data
    data = [("protein", sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    # Generate embeddings
    with torch.no_grad():
        if model_type == "esmc":
            # ESM-C models may have different output structure
            try:
                # Try standard ESM-C embedding extraction
                output = model(batch_tokens)

                # ESM-C models might return embeddings differently
                if hasattr(output, 'last_hidden_state'):
                    embeddings = output.last_hidden_state
                elif isinstance(output, dict) and 'representations' in output:
                    # If layer specified and available
                    if layer in output['representations']:
                        embeddings = output['representations'][layer]
                    else:
                        # Use last available layer
                        available_layers = list(output['representations'].keys())
                        embeddings = output['representations'][available_layers[-1]]
                elif isinstance(output, torch.Tensor):
                    embeddings = output
                else:
                    # Fallback to standard approach
                    results = model(batch_tokens, repr_layers=[layer] if layer else None)
                    embeddings = results["representations"][layer if layer else list(results["representations"].keys())[-1]]
            except Exception as e:
                tqdm.write(f"ESM-C embedding extraction fallback: {e}")
                # Fallback to ESM-2 style
                results = model(batch_tokens, repr_layers=[layer])
                embeddings = results["representations"][layer]
        else:
            # Standard ESM-1/ESM-2 embedding extraction
            results = model(batch_tokens, repr_layers=[layer])
            embeddings = results["representations"][layer]

    # Extract per-residue embeddings (remove special tokens <cls> and <eos>)
    sequence_embeddings = embeddings[0, 1:-1, :]  # Shape: (seq_length, embedding_dim)

    if pooling == "full":
        return sequence_embeddings.cpu().numpy()  # Shape: (seq_length, embedding_dim)

    if pooling.startswith("segments"):
        # Segment pooling: split sequence into N segments from C-terminal backwards,
        # mean-pool each segment. C-terminal segments are always clean/full-sized;
        # any remainder goes to the N-terminal (segment 1).
        n_segments = int(pooling.replace("segments", ""))
        L = sequence_embeddings.shape[0]
        seg_size = L // n_segments

        if seg_size == 0:
            # Protein shorter than n_segments: pad with zeros
            seg_means = []
            for i in range(n_segments):
                if i < L:
                    seg_means.append(sequence_embeddings[L - 1 - i, :])
                else:
                    seg_means.append(torch.zeros_like(sequence_embeddings[0, :]))
            seg_means.reverse()  # put N-terminal first
        else:
            # Split from C-terminal backwards
            seg_means = []
            end = L
            for i in range(n_segments - 1, 0, -1):
                start = end - seg_size
                seg_means.append(sequence_embeddings[start:end, :].mean(dim=0))
                end = start
            # Segment 1 (N-terminal) gets everything remaining
            seg_means.append(sequence_embeddings[0:end, :].mean(dim=0))
            seg_means.reverse()  # [seg1_Nterm, seg2, seg3, seg4_Cterm]

        return torch.cat(seg_means, dim=0).cpu().numpy()  # Shape: (n_segments * embedding_dim,)

    # Mean pool across sequence length to get single vector per protein
    mean_pooled_embedding = sequence_embeddings.mean(dim=0)  # Shape: (embedding_dim,)
    return mean_pooled_embedding.cpu().numpy()
def process_fasta_file(fasta_file, model, alphabet, device, layer, max_length=1024, model_name=None, key_by_md5=False, pooling="mean"):
    """
    Process all sequences in a FASTA file and generate embeddings.

    Args:
        fasta_file: Path to FASTA file
        model: ESM model
        alphabet: ESM alphabet
        device: torch device
        layer: Layer number for embeddings
        max_length: Maximum sequence length
        model_name: Name of the model (for type detection)
        key_by_md5: If True, key embeddings by MD5(sequence) instead of accession.
                    Sequences with the same MD5 are embedded only once.
    """
    embeddings_data = {}

    # Detect model type for proper embedding extraction
    model_type = detect_model_type(model_name) if model_name else None

    print(f"Processing FASTA file: {fasta_file}")
    if key_by_md5:
        print(f"Keying embeddings by MD5(sequence)")
    if model_type == "esmc":
        print(f"Using ESM-C model: embeddings from final layer")
    else:
        print(f"Extracting embeddings from layer: {layer}")

    # Read FASTA file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))
    total_sequences = len(sequences)

    print(f"Found {total_sequences} sequences")

    # If keying by MD5, pre-compute to skip duplicate sequences
    seen_md5s = set()
    skipped_duplicates = 0

    # Create progress bar
    progress_bar = tqdm(sequences, desc="Generating embeddings", unit="seq")

    for record in progress_bar:
        accession = record.id
        sequence = str(record.seq)

        if key_by_md5:
            md5 = hashlib.md5(sequence.encode()).hexdigest()
            if md5 in seen_md5s:
                skipped_duplicates += 1
                continue
            seen_md5s.add(md5)
            key = md5
        else:
            key = accession

        # Update progress bar description with current accession
        progress_bar.set_postfix({"Current": accession[:20] + "..." if len(accession) > 20 else accession,
                                 "Length": len(sequence)})

        try:
            embedding = get_protein_embedding(sequence, model, alphabet, device, layer, max_length, model_type, pooling)
            embeddings_data[key] = embedding
        except Exception as e:
            tqdm.write(f"Error processing {accession}: {str(e)}")
            continue

    progress_bar.close()
    print(f"Successfully processed {len(embeddings_data)}/{total_sequences} sequences")
    if key_by_md5:
        print(f"Skipped {skipped_duplicates} duplicate sequences (same MD5)")

    return embeddings_data

    """
    Save embeddings to CSV file with specified column format.
    """
    if not embeddings_data:
        print("No embeddings to save!")
        return

    # Get embedding dimension from first embedding
    first_embedding = next(iter(embeddings_data.values()))
    embedding_dim = len(first_embedding)

    print(f"Embedding dimension: {embedding_dim}")

    # Create column names: accession + numbered columns (0 to embedding_dim-1)
    columns = ['accession'] + [str(i) for i in range(embedding_dim)]

    # Prepare data for DataFrame
    rows = []
    for accession, embedding in embeddings_data.items():
        row = [accession] + embedding.tolist()
        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=columns)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Embeddings saved to: {output_file}")
    print(f"Shape: {df.shape}")


def save_embeddings_to_npz(embeddings_data, output_file):
    """
    Save per-residue embeddings to NPZ file.
    Each key is the sequence ID, value is (seq_length, 1280) array.
    """
    if not embeddings_data:
        print("No embeddings to save!")
        return

    # Convert to dict of numpy arrays
    embeddings_dict = {seq_id: emb for seq_id, emb in embeddings_data.items()}

    # Save to NPZ
    np.savez_compressed(output_file, **embeddings_dict)

    print(f"Embeddings saved to: {output_file}")
    print(f"Number of sequences: {len(embeddings_dict)}")

    # Print example shape
    first_id = list(embeddings_dict.keys())[0]
    print(f"Example shape ({first_id}): {embeddings_dict[first_id].shape}")

def main():
    parser = argparse.ArgumentParser(
        description="Generate ESM-2 embeddings for protein sequences in FASTA format"
    )
    parser.add_argument("input_fasta", help="Input FASTA file")
    parser.add_argument("output_csv", help="Output CSV file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to local ESM-2 checkpoint file (.pt)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="ESM model name (e.g., esm2_t33_650M_UR50D, esm2_t36_3B_UR50D, esm2_t48_15B_UR50D)"
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer number to extract embeddings from (default: last layer). ESM2-650M has 33 layers (0-32)."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum sequence length (default: auto-detected based on model)"
    )
    parser.add_argument(
        "--use_extended",
        action="store_true",
        help="Use extended max length (2048) for compatible models (requires more GPU memory)"
    )
    parser.add_argument(
        "--analyze_only",
        action="store_true",
        help="Only analyze sequence lengths, don't generate embeddings"
    )
    parser.add_argument(
        "--key_by_md5",
        action="store_true",
        help="Key embeddings by MD5(sequence) instead of FASTA accession. "
             "Duplicate sequences are embedded only once."
    )
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        help="Pooling strategy: 'mean' (default, 1D vector), "
             "'full' (per-residue L×D matrix), or "
             "'segments4' (4 segment means from C-terminal, 4×D vector). "
             "Use 'segmentsN' for N segments."
    )

    args = parser.parse_args()

    # Validate input file
    if not Path(args.input_fasta).exists():
        print(f"Error: Input file {args.input_fasta} does not exist!")
        sys.exit(1)

    # Validate checkpoint file if provided
    if args.checkpoint and not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint file {args.checkpoint} does not exist!")
        sys.exit(1)

    # Determine model name
    if args.model:
        model_name = args.model
    elif args.checkpoint:
        # Try to infer model from checkpoint name if not specified
        checkpoint_name = Path(args.checkpoint).stem.lower()
        if "esmc" in checkpoint_name:
            if "6b" in checkpoint_name:
                model_name = "esmc_6b"
            elif "600" in checkpoint_name:
                model_name = "esmc_600m"
            elif "300" in checkpoint_name:
                model_name = "esmc_300m"
            else:
                model_name = "esmc_600m"  # Default ESM-C
            print(f"Detected ESM-C model from checkpoint: {model_name}")
        elif "15b" in checkpoint_name or "t48" in checkpoint_name:
            model_name = "esm2_t48_15B_UR50D"
        elif "3b" in checkpoint_name or "t36" in checkpoint_name:
            model_name = "esm2_t36_3B_UR50D"
        elif "650m" in checkpoint_name or "t33" in checkpoint_name:
            model_name = "esm2_t33_650M_UR50D"
        elif "150m" in checkpoint_name or "t30" in checkpoint_name:
            model_name = "esm2_t30_150M_UR50D"
        else:
            model_name = "esm2_t33_650M_UR50D"  # Default
            print(f"Could not infer model from checkpoint name, using default: {model_name}")
    else:
        # Default model for pretrained
        model_name = "esm2_t48_15B_UR50D"

    print(f"Model selected: {model_name}")


    # Determine max length
    if args.max_length is not None:
        # User specified max length
        max_length = args.max_length
        print(f"Using user-specified max length: {max_length}")
    else:
        # Auto-detect based on model
        max_length = get_max_length_for_model(model_name, args.use_extended)

    # Analyze sequence lengths
    print("\n" + "="*60)
    print("SEQUENCE LENGTH ANALYSIS")
    print("="*60)
    stats = analyze_sequence_lengths(args.input_fasta, max_length)
    print(f"Total sequences: {stats['total_sequences']}")
    print(f"Max length in dataset: {stats['max_length']}")
    print(f"Mean length: {stats['mean_length']:.1f}")
    print(f"Median length: {stats['median_length']:.1f}")
    print(f"\nUsing max length: {max_length}")
    print(f"Sequences to be truncated: {stats['truncated_count']} ({stats['truncated_percentage']:.1f}%)")

    if stats['truncated_count'] > 0:
        print(f"Max residues to be lost: {stats['max_residues_lost']:.0f}")
        print(f"Mean residues to be lost: {stats['mean_residues_lost']:.1f}")

        if stats['truncated_percentage'] > 20:
            print("\n⚠️  WARNING: >20% of sequences will be truncated!")
            if not args.use_extended and model_name in ESM_MODEL_EXTENDED_LENGTHS:
                print(f"   Consider using --use_extended for max length of {ESM_MODEL_EXTENDED_LENGTHS[model_name]}")
            if model_name != "esm2_t48_15B_UR50D":
                print("   Consider using a larger model (e.g., esm2_t36_3B_UR50D or esm2_t48_15B_UR50D)")

    print("="*60)

    # If analyze only, stop here
    if args.analyze_only:
        print("\nAnalysis complete (--analyze_only flag set)")
        sys.exit(0)

    # Load model
    try:
        if args.checkpoint:
            model, alphabet, device, loaded_model_name = load_esm2_model_with_checkpoint(args.checkpoint, model_name)
        else:
            model, alphabet, device, loaded_model_name = load_esm2_model_pretrained(model_name)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)

    # Determine layer to use
    model_type = detect_model_type(loaded_model_name)

    if model_type == "esmc":
        # ESM-C models typically use the final layer
        if hasattr(model, 'num_layers'):
            layer = model.num_layers if args.layer is None else args.layer
        else:
            layer = None  # ESM-C may not have traditional layer structure
        print(f"ESM-C model detected - using final representations")
    else:
        if args.layer is None:
            layer = model.num_layers
            print(f"No layer specified, using last layer: {layer}")
        else:
            layer = args.layer
            # Validate layer number
            if layer < 0 or layer > model.num_layers:
                print(f"Error: Layer {layer} is out of range. Model has layers 0-{model.num_layers}")
                sys.exit(1)

    # Process FASTA file
    try:
        embeddings_data = process_fasta_file(args.input_fasta, model, alphabet, device, layer, max_length, loaded_model_name, key_by_md5=args.key_by_md5, pooling=args.pooling)
    except Exception as e:
        print(f"Error processing FASTA file: {str(e)}")
        sys.exit(1)

    # Save results
    try:
        output_npz = args.output_csv.replace('.csv', '.npz')
        save_embeddings_to_npz(embeddings_data, output_npz)
    except Exception as e:
        print(f"Error saving results: {str(e)}")
        sys.exit(1)

    print("Done!")

if __name__ == "__main__":
    main()

 
