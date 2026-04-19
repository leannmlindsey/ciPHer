"""Load phage-host interaction matrices and protein mappings."""

import csv
import os
from collections import defaultdict


# Valid dataset names
DATASETS = ['CHEN', 'GORODNICHIV', 'UCSD', 'PBIP', 'PhageHostLearn']


def load_interaction_matrix(dataset_dir, dataset_name=None):
    """Load a standardized interaction matrix.

    Expects the ciPHer standardized format at:
        {dataset_dir}/metadata/interaction_matrix.tsv

    Columns: host_id, host_assembly, host_K, host_O, phage_id, label

    Args:
        dataset_dir: Path to the dataset directory (e.g., HOST_RANGE/PBIP/)
        dataset_name: Optional, for logging only.

    Returns:
        dict: {phage_id: {host_id: 0 or 1}}
    """
    path = os.path.join(dataset_dir, 'metadata', 'interaction_matrix.tsv')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Standardized interaction matrix not found: {path}\n"
            f"Run data standardization first.")

    interactions = defaultdict(dict)
    with open(path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            host = row['host_id'].strip()
            phage = row['phage_id'].strip()
            label = row['label'].strip()
            interactions[phage][host] = 1 if label in ('1', '1.0') else 0

    return dict(interactions)


def load_interaction_pairs(dataset_dir):
    """Load interaction matrix as a list of enriched pairs.

    Returns all pairs with serotype information attached.

    Args:
        dataset_dir: Path to dataset directory.

    Returns:
        list of dicts, each with: host_id, host_assembly, host_K, host_O,
        phage_id, label
    """
    path = os.path.join(dataset_dir, 'metadata', 'interaction_matrix.tsv')
    pairs = []
    with open(path) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            pairs.append({
                'host_id': row['host_id'].strip(),
                'host_assembly': row['host_assembly'].strip(),
                'host_K': row['host_K'].strip(),
                'host_O': row['host_O'].strip(),
                'phage_id': row['phage_id'].strip(),
                'label': int(float(row['label'].strip())),
            })
    return pairs


def load_phage_protein_mapping(filepath):
    """Load phage-to-protein mapping from a CSV file.

    Columns: phage_id (or matrix_phage_name), protein_id

    Args:
        filepath: Path to phage_protein_mapping.csv

    Returns:
        dict: {phage_id: set of protein_ids}
    """
    mapping = defaultdict(set)
    with open(filepath) as f:
        reader = csv.DictReader(f)
        # Handle both old column name and new
        phage_col = 'phage_id' if 'phage_id' in reader.fieldnames else 'matrix_phage_name'
        for row in reader:
            phage = row[phage_col].strip()
            protein = row['protein_id'].strip()
            mapping[phage].add(protein)
    return dict(mapping)


def load_training_map(filepath):
    """Load the standardized training host-phage-protein map.

    Expects columns: host_assembly, host_K, host_O, phage_id,
                     protein_id, is_tsp, protein_md5

    Args:
        filepath: Path to host_phage_protein_map.tsv

    Returns:
        list of dicts with all columns
    """
    rows = []
    with open(filepath) as f:
        for row in csv.DictReader(f, delimiter='\t'):
            rows.append({
                'host_assembly': row['host_assembly'].strip(),
                'host_K': row['host_K'].strip(),
                'host_O': row['host_O'].strip(),
                'phage_id': row['phage_id'].strip(),
                'protein_id': row['protein_id'].strip(),
                'is_tsp': int(row['is_tsp'].strip()),
                'protein_md5': row['protein_md5'].strip(),
            })
    return rows
