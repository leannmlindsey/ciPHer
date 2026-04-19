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
