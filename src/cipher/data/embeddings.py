"""Load protein embeddings from NPZ files (keyed by MD5 hash)."""

import numpy as np


def load_embeddings(filepath, md5_filter=None):
    """Load embeddings from an NPZ file.

    Args:
        filepath: Path to .npz file (keys are MD5 hashes, values are vectors).
        md5_filter: Optional set of MD5s to load. If None, loads all.
                    Use this for large files to reduce memory.

    Returns:
        dict: {md5_hash: numpy_array}
    """
    data = np.load(filepath)
    if md5_filter is None:
        return {k: data[k] for k in data.files}

    md5_filter = set(md5_filter)
    return {k: data[k] for k in data.files if k in md5_filter}


def load_embeddings_concat(filepath_1, filepath_2, md5_filter=None, log=print):
    """Load two NPZ embedding files and concatenate them per MD5.

    Feature order is [file_1, file_2] (e.g. pLM first, k-mer second), but
    downstream models treat the result as a single vector so order is
    purely a labeling convention.

    Every MD5 passing `md5_filter` must be present in BOTH files — a
    missing key raises ValueError so data gaps are surfaced rather than
    silently reducing the training set.

    Args:
        filepath_1: Path to first NPZ (e.g. pLM embedding).
        filepath_2: Path to second NPZ (e.g. k-mer features).
        md5_filter: Optional set of MD5s to load. If None, uses the
                    intersection of keys present in both files.
        log: Callable for progress messages (default: print). Pass a
             no-op lambda to silence.

    Returns:
        dict: {md5_hash: concatenated_vector (file_1 dims + file_2 dims)}
    """
    d1 = load_embeddings(filepath_1, md5_filter=md5_filter)
    d2 = load_embeddings(filepath_2, md5_filter=md5_filter)

    if md5_filter is None:
        target = set(d1.keys()) & set(d2.keys())
    else:
        target = set(md5_filter)

    missing_in_1 = target - d1.keys()
    missing_in_2 = target - d2.keys()
    if missing_in_1 or missing_in_2:
        # Show a few examples so the error is actionable
        sample_1 = list(missing_in_1)[:5]
        sample_2 = list(missing_in_2)[:5]
        raise ValueError(
            f'Embedding coverage mismatch: '
            f'{len(missing_in_1)} MD5s missing from {filepath_1} '
            f'(e.g. {sample_1}); '
            f'{len(missing_in_2)} missing from {filepath_2} '
            f'(e.g. {sample_2}). '
            f'Both files must cover every MD5 in the training set.')

    dim_1 = next(iter(d1.values())).shape[-1] if d1 else 0
    dim_2 = next(iter(d2.values())).shape[-1] if d2 else 0
    log(f'  concat embeddings: {dim_1} + {dim_2} = {dim_1 + dim_2} dims '
        f'({len(target):,} MD5s)')

    return {m: np.concatenate([d1[m], d2[m]]) for m in target}
