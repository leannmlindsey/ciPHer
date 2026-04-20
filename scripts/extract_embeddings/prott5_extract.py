#!/usr/bin/env python3
"""
Generate ProtT5 embeddings for protein sequences.

Uses any Rostlab ProtT5 model (--model_name). By default pools per-residue
representations with mean pooling to produce a (embedding_dim,) vector per
protein. Alternative pooling via --pooling segmentsN splits the sequence
into N equal segments (from the C-terminal), mean-pools each, and
concatenates to (N * embedding_dim,).

Supports --key_by_md5 to key embeddings by MD5(sequence) for compatibility
with the training pipeline.

Requirements:
    pip install torch transformers sentencepiece

Usage:
    python prott5_extract.py input.faa output.npz --key_by_md5
    python prott5_extract.py input.faa output.npz --key_by_md5 --pooling segments4
    python prott5_extract.py input.faa output.npz --model_name Rostlab/prot_t5_xxl_uniref50 --half_precision
"""

import argparse
import hashlib
import os
import sys

import numpy as np
import torch
from pathlib import Path

try:
    from transformers import T5Tokenizer, T5EncoderModel
except ImportError:
    sys.exit("ERROR: transformers required. Install with: pip install transformers sentencepiece")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x


def pool_residue_embeddings(residue_embeddings, pooling):
    """Pool a (L, D) tensor down to either (D,) or (N*D,).

    Supported pooling:
        'mean'       -> mean across the length axis, returns (D,)
        'segmentsN'  -> split into N equal segments (remainder to the
                        N-terminal), mean-pool each, concat to (N*D,).
                        For sequences shorter than N, missing slots are
                        zero-padded so the output dim is invariant.
    """
    if pooling == 'mean':
        return residue_embeddings.mean(dim=0)

    if pooling.startswith('segments'):
        n_segments = int(pooling.replace('segments', ''))
        L = residue_embeddings.shape[0]
        seg_size = L // n_segments

        if seg_size == 0:
            # Sequence shorter than n_segments: fall back to per-residue
            # entries for what we have, zero-pad the rest.
            seg_means = []
            for i in range(n_segments):
                if i < L:
                    seg_means.append(residue_embeddings[L - 1 - i, :])
                else:
                    seg_means.append(torch.zeros_like(residue_embeddings[0, :]))
            seg_means.reverse()  # N-terminal first
            return torch.cat(seg_means, dim=0)

        seg_means = []
        end = L
        for i in range(n_segments - 1, 0, -1):
            start = end - seg_size
            seg_means.append(residue_embeddings[start:end, :].mean(dim=0))
            end = start
        # Segment 1 (N-terminal) gets everything remaining
        seg_means.append(residue_embeddings[0:end, :].mean(dim=0))
        seg_means.reverse()  # [seg1_Nterm, ..., segN_Cterm]
        return torch.cat(seg_means, dim=0)

    raise ValueError(
        f"Unknown pooling: {pooling!r}. Use 'mean' or 'segmentsN' (any N>=2).")


def parse_fasta(filepath):
    """Parse FASTA file. Returns list of (id, sequence)."""
    sequences = []
    current_id = None
    current_seq = []
    with open(filepath) as f:
        for line in f:
            if line.startswith(">"):
                if current_id:
                    sequences.append((current_id, "".join(current_seq)))
                current_id = line[1:].strip().split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_id:
            sequences.append((current_id, "".join(current_seq)))
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Generate ProtT5 embeddings")
    parser.add_argument("input_fasta", help="Input FASTA file")
    parser.add_argument("output_npz", help="Output NPZ file")
    parser.add_argument("--model_name", default="Rostlab/prot_t5_xl_uniref50",
                        help="ProtT5 model name (default: Rostlab/prot_t5_xl_uniref50)")
    parser.add_argument("--max_length", type=int, default=None,
                        help="Maximum sequence length (default: no limit, truncate to model max)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for inference (default: 4, reduce if OOM)")
    parser.add_argument("--key_by_md5", action="store_true",
                        help="Key embeddings by MD5(sequence) instead of FASTA accession")
    parser.add_argument("--half_precision", action="store_true",
                        help="Use float16 for inference (saves GPU memory)")
    parser.add_argument("--pooling", default="mean",
                        help="Pooling strategy: 'mean' (default, D-dim output) "
                             "or 'segmentsN' for any N>=2 (N*D-dim output). "
                             "Example: --pooling segments4")
    args = parser.parse_args()

    # Validate pooling early
    if args.pooling != 'mean' and not args.pooling.startswith('segments'):
        sys.exit(f"ERROR: invalid --pooling {args.pooling!r}. Use 'mean' or 'segmentsN'.")

    if not Path(args.input_fasta).exists():
        sys.exit(f"ERROR: {args.input_fasta} not found")

    # Set cache directory
    cache_dir = os.environ.get("TORCH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "torch"))
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = cache_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    print(f"Loading ProtT5 model: {args.model_name}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Device: {device}")

    tokenizer = T5Tokenizer.from_pretrained(args.model_name, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(args.model_name)

    if args.half_precision:
        model = model.half()
        print("  Using float16 precision")

    model = model.to(device)
    model.eval()

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Parse FASTA
    print(f"\nReading {args.input_fasta}...")
    sequences = parse_fasta(args.input_fasta)
    print(f"  Found {len(sequences):,} sequences")

    # Sort by length for efficient batching
    sequences.sort(key=lambda x: len(x[1]))

    # Analyze lengths
    lengths = [len(s) for _, s in sequences]
    print(f"  Length range: {min(lengths)} - {max(lengths)}")
    print(f"  Mean length: {np.mean(lengths):.0f}")

    if args.max_length:
        truncated = sum(1 for l in lengths if l > args.max_length)
        print(f"  Will truncate: {truncated:,} sequences (>{args.max_length} aa)")

    # Process in batches
    print(f"\nGenerating embeddings (batch_size={args.batch_size})...")
    embeddings_dict = {}
    seen_md5s = set()
    skipped_duplicates = 0

    # ProtT5 expects spaces between amino acids
    def prepare_sequence(seq, max_length=None):
        if max_length:
            seq = seq[:max_length]
        # Replace uncommon AAs and add spaces
        seq = seq.replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")
        return " ".join(list(seq))

    n_batches = (len(sequences) + args.batch_size - 1) // args.batch_size
    pbar = tqdm(range(0, len(sequences), args.batch_size),
                desc="Embedding", unit="batch", total=n_batches)

    for batch_start in pbar:
        batch = sequences[batch_start:batch_start + args.batch_size]
        batch_ids = []
        batch_seqs = []
        batch_keys = []

        for seq_id, seq in batch:
            if args.key_by_md5:
                md5 = hashlib.md5(seq.encode()).hexdigest()
                if md5 in seen_md5s:
                    skipped_duplicates += 1
                    continue
                seen_md5s.add(md5)
                key = md5
            else:
                key = seq_id

            batch_ids.append(seq_id)
            batch_seqs.append(prepare_sequence(seq, args.max_length))
            batch_keys.append(key)

        if not batch_seqs:
            continue

        # Tokenize
        ids = tokenizer.batch_encode_plus(
            batch_seqs,
            add_special_tokens=True,
            padding="longest",
            return_tensors="pt",
        )

        input_ids = ids["input_ids"].to(device)
        attention_mask = ids["attention_mask"].to(device)

        # Generate embeddings
        with torch.no_grad():
            results = model(input_ids=input_ids, attention_mask=attention_mask)
            # results.last_hidden_state: (batch, seq_len, 1024)
            hidden_states = results.last_hidden_state

        # Pool (excluding padding + EOS)
        for i in range(len(batch_keys)):
            seq_len = attention_mask[i].sum().item()
            # Exclude special tokens: take [0:seq_len-1] to skip EOS
            residues = hidden_states[i, :int(seq_len) - 1, :]
            emb = pool_residue_embeddings(residues, args.pooling)
            if args.half_precision:
                emb = emb.float()  # convert back to float32 for storage
            embeddings_dict[batch_keys[i]] = emb.cpu().numpy()

        # Update progress bar
        pbar.set_postfix({
            "embedded": len(embeddings_dict),
            "current_len": len(batch[0][1]),
        })

    pbar.close()

    print(f"\nResults:")
    print(f"  Embedded: {len(embeddings_dict):,} sequences")
    if args.key_by_md5:
        print(f"  Skipped duplicates: {skipped_duplicates:,}")

    # Verify dimensions
    first_key = next(iter(embeddings_dict))
    emb_dim = embeddings_dict[first_key].shape[0]
    print(f"  Embedding dimension: {emb_dim}")

    # Save
    print(f"\nSaving to {args.output_npz}...")
    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez_compressed(args.output_npz, **embeddings_dict)

    # Verify
    verify = np.load(args.output_npz)
    print(f"  Verified: {len(verify.files):,} embeddings, shape={verify[verify.files[0]].shape}")
    print("Done!")


if __name__ == "__main__":
    main()
