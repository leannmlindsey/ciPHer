"""Protein sequence utilities: FASTA parsing, MD5 hashing, annotation loading."""

import csv
import hashlib
from collections import defaultdict


def compute_md5(sequence):
    """Compute MD5 hash of a protein sequence string."""
    return hashlib.md5(sequence.encode()).hexdigest()


def load_fasta(filepath, id_filter=None):
    """Parse a FASTA file, return {protein_id: sequence}.

    Args:
        filepath: Path to FASTA file.
        id_filter: Optional set of protein IDs to keep. If None, loads all.

    Returns:
        dict: {protein_id: sequence_string}
    """
    proteins = {}
    current_id = None
    current_seq = []

    with open(filepath) as f:
        for line in f:
            if line.startswith('>'):
                if current_id is not None:
                    seq = ''.join(current_seq)
                    if id_filter is None or current_id in id_filter:
                        proteins[current_id] = seq
                current_id = line[1:].strip().split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())

    if current_id is not None:
        seq = ''.join(current_seq)
        if id_filter is None or current_id in id_filter:
            proteins[current_id] = seq

    return proteins


def load_fasta_md5(filepath, id_filter=None):
    """Parse FASTA and return {protein_id: md5_hash}.

    Convenience wrapper that computes MD5 for each sequence.

    Args:
        filepath: Path to FASTA file.
        id_filter: Optional set of protein IDs to keep.

    Returns:
        dict: {protein_id: md5_hash_string}
    """
    proteins = load_fasta(filepath, id_filter=id_filter)
    return {pid: compute_md5(seq) for pid, seq in proteins.items()}


def load_glycan_binders(filepath):
    """Load glycan binder annotations TSV.

    Args:
        filepath: Path to glycan_binders_custom.tsv

    Returns:
        dict: {protein_id: {tool_name: value, total_sources: int, ...}}
    """
    binders = {}
    with open(filepath) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            binders[row['protein_id']] = dict(row)
    return binders


def load_positive_list(filepath):
    """Load a list of positive protein IDs (one per line).

    Args:
        filepath: Path to pipeline_positive.list

    Returns:
        set of protein ID strings
    """
    with open(filepath) as f:
        return {line.strip() for line in f if line.strip()}


def load_phage_protein_pairs(filepath):
    """Load phage-protein mapping TSV.

    Expects tab-separated file where each row links a phage genome
    to a protein ID. The phage genome ID encodes the host accession
    (extractable via extract_host_accession).

    Args:
        filepath: Path to phage_protein_positives.tsv

    Returns:
        list of (phage_genome_id, protein_id) tuples
    """
    pairs = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def extract_host_accession(phage_genome_id):
    """Extract host GCA/GCF accession from a phage genome ID.

    The phage genome ID format embeds the host accession, typically
    as the first underscore-separated component or similar pattern.

    Args:
        phage_genome_id: String like 'GCA_000001_phage_123'

    Returns:
        Host accession string, or None if not extractable.
    """
    parts = phage_genome_id.split('_')
    if len(parts) >= 2 and parts[0] in ('GCA', 'GCF'):
        return f"{parts[0]}_{parts[1]}"
    # Try first two underscore-joined parts
    if len(parts) >= 3:
        candidate = f"{parts[0]}_{parts[1]}"
        if candidate.startswith(('GCA', 'GCF')):
            return candidate
    return None
