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
    parser.add_argument("--checkpoint_every", type=int, default=2000,
                        help="Save a .partial.npz checkpoint every N sequences "
                             "so a late crash doesn't lose hours of work. "
                             "On restart, existing keys in the partial file "
                             "are skipped. Set to 0 to disable (default: 2000).")
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

    # Resume support: load any pre-existing partial checkpoint and skip
    # keys we already have.
    partial_path = args.output_npz + ".partial.npz"
    embeddings_dict = {}
    if os.path.exists(partial_path):
        print(f"  Resuming from checkpoint: {partial_path}")
        prev = np.load(partial_path)
        for k in prev.files:
            embeddings_dict[k] = prev[k]
        print(f"  Loaded {len(embeddings_dict):,} pre-existing embeddings")

    seen_md5s = set(embeddings_dict.keys()) if args.key_by_md5 else set()
    skipped_duplicates = 0
    since_last_checkpoint = 0
    oom_skipped = 0

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

        # Skip anything already in the (resumed) dict.
        pending = [(sid, seq, key) for sid, seq, key in zip(batch_ids, batch_seqs, batch_keys)
                   if key not in embeddings_dict]
        if not pending:
            continue
        cur_ids = [p[0] for p in pending]
        cur_seqs = [p[1] for p in pending]
        cur_keys = [p[2] for p in pending]

        # Encode with an OOM-aware retry ladder: full batch → batch_size 1 → skip.
        def _run(seqs):
            ids = tokenizer.batch_encode_plus(
                seqs, add_special_tokens=True, padding="longest",
                return_tensors="pt")
            input_ids = ids["input_ids"].to(device)
            attention_mask = ids["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            return out.last_hidden_state, attention_mask

        try:
            hidden_states, attention_mask = _run(cur_seqs)
            process_seqs = cur_seqs
            process_keys = cur_keys
        except torch.cuda.OutOfMemoryError:
            # Batch OOM — fall back to per-sequence, skipping any that still OOM.
            torch.cuda.empty_cache()
            hidden_states = None  # signal fallback path below
            fallback_results = []
            for sid, seq, key in zip(cur_ids, cur_seqs, cur_keys):
                try:
                    h, am = _run([seq])
                    fallback_results.append((key, h[0], am[0]))
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"\n  OOM on single sequence ({sid}, len={len(seq.replace(' ',''))}); skipping",
                          file=sys.stderr)
                    oom_skipped += 1
            # Pool directly from the fallback list
            for key, h, am in fallback_results:
                seq_len = int(am.sum().item())
                residues = h[:seq_len - 1, :]
                emb = pool_residue_embeddings(residues, args.pooling)
                if args.half_precision:
                    emb = emb.float()
                embeddings_dict[key] = emb.cpu().numpy()
                since_last_checkpoint += 1

        if hidden_states is not None:
            # Normal path: pool full batch
            for i in range(len(cur_keys)):
                seq_len = attention_mask[i].sum().item()
                residues = hidden_states[i, :int(seq_len) - 1, :]
                emb = pool_residue_embeddings(residues, args.pooling)
                if args.half_precision:
                    emb = emb.float()
                embeddings_dict[cur_keys[i]] = emb.cpu().numpy()
                since_last_checkpoint += 1

        # Periodic checkpoint write
        if args.checkpoint_every and since_last_checkpoint >= args.checkpoint_every:
            os.makedirs(os.path.dirname(partial_path) or ".", exist_ok=True)
            np.savez(partial_path, **embeddings_dict)
            since_last_checkpoint = 0

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
    if oom_skipped:
        print(f"  OOM-skipped sequences: {oom_skipped:,}")

    # Verify dimensions
    first_key = next(iter(embeddings_dict))
    emb_dim = embeddings_dict[first_key].shape[0]
    print(f"  Embedding dimension: {emb_dim}")

    # Save
    print(f"\nSaving to {args.output_npz}...")
    os.makedirs(os.path.dirname(args.output_npz) or ".", exist_ok=True)
    np.savez_compressed(args.output_npz, **embeddings_dict)

    # Remove partial checkpoint now that the final output is written.
    if os.path.exists(partial_path):
        os.remove(partial_path)

    # Verify
    verify = np.load(args.output_npz)
    print(f"  Verified: {len(verify.files):,} embeddings, shape={verify[verify.files[0]].shape}")
    print("Done!")


if __name__ == "__main__":
    main()
