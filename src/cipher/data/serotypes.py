"""Load and normalize serotype annotations."""

import csv

NULL_LABELS = {'null', 'N/A', 'Unknown', '', 'NA', 'n/a'}


def load_serotypes(filepath):
    """Load host serotype annotations from TSV.

    Handles UTF-8 BOM and carriage returns in the file.

    Args:
        filepath: Path to serotypes TSV (columns: genome_id, K, O)

    Returns:
        dict: {genome_id: {'K': k_type, 'O': o_type}}
    """
    serotypes = {}
    with open(filepath, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            gid = row['genome_id'].strip().rstrip('\r')
            k = row['K'].strip().rstrip('\r')
            o = row['O'].strip().rstrip('\r')
            serotypes[gid] = {'K': k, 'O': o}
    return serotypes


def is_null(serotype_value):
    """Check if a serotype value is null/missing."""
    return serotype_value in NULL_LABELS


def normalize_k_type(k_type):
    """Normalize K-type naming between conventions.

    Some sources use 'K1', others use 'KL1'. This converts
    K→KL format for consistency with Kaptive output.

    Args:
        k_type: String like 'K1', 'KL1', 'KL107'

    Returns:
        Normalized K-type string in KL format.
    """
    if is_null(k_type):
        return k_type
    if k_type.startswith('KL'):
        return k_type
    if k_type.startswith('K'):
        return 'KL' + k_type[1:]
    return k_type


def normalize_k_to_short(k_type):
    """Convert KL format back to short K format.

    KL1→K1, KL23→K23. KL107+ stay as-is (no short equivalent).

    Args:
        k_type: String like 'KL1', 'KL107'

    Returns:
        Short K-type string where possible.
    """
    if is_null(k_type):
        return k_type
    if k_type.startswith('KL'):
        num = k_type[2:]
        try:
            n = int(num)
            if n <= 82:
                return f'K{n}'
        except ValueError:
            pass
    return k_type
