"""Bio-motivated host/phage ranking functions.

Scoring model: host has 2 locks (K, O), phage has N keys (proteins).
Score = max over proteins of max(P(K_host), P(O_host)).
If any key fits any lock, the phage can infect.

Two ranking modes:
    - rank_hosts: given a phage, rank candidate hosts
    - rank_phages: given a host, rank candidate phages

evaluate_rankings() runs both modes for a validation dataset and
returns HR@k curves and MRR.
"""

import os
from collections import defaultdict


def rank_hosts(predictor, phage_proteins, candidate_hosts, serotypes,
               emb_dict, pid_md5):
    """Given a phage, rank all candidate hosts by predicted interaction score.

    Args:
        predictor: Predictor instance
        phage_proteins: set of protein IDs for this phage
        candidate_hosts: list of host IDs to rank
        serotypes: dict {host_id: {'K': ..., 'O': ...}}
        emb_dict: dict {md5: embedding_array}
        pid_md5: dict {protein_id: md5_hash}

    Returns:
        list of (host_id, score) sorted by score descending
    """
    prot_md5s = [pid_md5[p] for p in phage_proteins if p in pid_md5]

    if not prot_md5s:
        return []

    host_scores = []
    for host in candidate_hosts:
        if host not in serotypes:
            continue
        host_k = serotypes[host]['K']
        host_o = serotypes[host]['O']
        # score_phage_md5s defers to score_pair for single-embedding
        # predictors; dual-embedding predictors override it to look up
        # K and O embeddings from their own per-head dicts.
        score = predictor.score_phage_md5s(prot_md5s, host_k, host_o, emb_dict)
        if score is not None:
            host_scores.append((host, score))

    host_scores.sort(key=lambda x: -x[1])
    return host_scores


def rank_phages(predictor, host_id, candidate_phages, phage_protein_map,
                serotypes, emb_dict, pid_md5):
    """Given a host, rank all candidate phages by predicted interaction score.

    Args:
        predictor: Predictor instance
        host_id: the host to score against
        candidate_phages: list of phage IDs to rank
        phage_protein_map: dict {phage: set of protein_ids}
        serotypes: dict {host_id: {'K': ..., 'O': ...}}
        emb_dict: dict {md5: embedding_array}
        pid_md5: dict {protein_id: md5_hash}

    Returns:
        list of (phage_id, score) sorted by score descending
    """
    if host_id not in serotypes:
        return []

    host_k = serotypes[host_id]['K']
    host_o = serotypes[host_id]['O']

    phage_scores = []
    for phage in candidate_phages:
        proteins = phage_protein_map.get(phage, set())
        prot_md5s = [pid_md5[p] for p in proteins if p in pid_md5]

        score = predictor.score_phage_md5s(prot_md5s, host_k, host_o, emb_dict)
        if score is not None:
            phage_scores.append((phage, score))

    phage_scores.sort(key=lambda x: -x[1])
    return phage_scores


def _ranks_with_ties(scored_items, tie_method='competition'):
    """Compute ranks for a sorted-descending list of (item, score) tuples.

    Args:
        scored_items: list of (item, score), already sorted by score descending
        tie_method: how to handle ties.
            'competition' (default): all tied items get the best rank in the group.
                e.g. scores [9, 9, 9, 5, 1] -> ranks [1, 1, 1, 4, 5]
            'arbitrary': ranks by position (preserves insertion order on ties).
                e.g. scores [9, 9, 9, 5, 1] -> ranks [1, 2, 3, 4, 5]

    Returns:
        dict {item: rank}
    """
    item_to_rank = {}
    if tie_method == 'arbitrary':
        for i, (item, _) in enumerate(scored_items):
            item_to_rank[item] = i + 1
        return item_to_rank

    # Competition ranking
    current_rank = 0
    prev_score = None
    for i, (item, score) in enumerate(scored_items):
        if score != prev_score:
            current_rank = i + 1
            prev_score = score
        item_to_rank[item] = current_rank
    return item_to_rank


def evaluate_rankings(predictor, dataset_name, dataset_dir,
                      emb_dict, pid_md5, max_k=20, tie_method='competition'):
    """Run full evaluation for one validation dataset: both ranking directions.

    Builds serotypes from the interaction matrix (which has host_K, host_O
    inline), so no external serotype file is needed.

    Args:
        predictor: Predictor instance
        dataset_name: e.g., 'PBIP' (for logging only)
        dataset_dir: path to HOST_RANGE/{dataset}/
        emb_dict: dict {md5: embedding}
        pid_md5: dict {protein_id: md5}
        max_k: maximum k for HR@k
        tie_method: 'competition' (default) gives all tied items the best rank
                    in their group — appropriate when the model can't actually
                    discriminate between items with the same predicted serotype.
                    'arbitrary' uses positional ranking (old behavior).

    Returns:
        dict with 'rank_hosts' and 'rank_phages' results, each containing:
            n_pairs, n_phages, n_hosts, hr_at_k, mrr
    """
    from cipher.data.interactions import (
        load_interaction_pairs, load_phage_protein_mapping,
    )
    from cipher.evaluation.metrics import hr_curve, mrr

    # Load interaction data and build serotypes from it
    pairs = load_interaction_pairs(dataset_dir)
    pm_path = os.path.join(dataset_dir, 'metadata', 'phage_protein_mapping.csv')
    phage_protein_map = load_phage_protein_mapping(pm_path)

    # Build interactions dict and serotypes from the pairs
    interactions = defaultdict(dict)
    serotypes = {}
    for p in pairs:
        interactions[p['phage_id']][p['host_id']] = p['label']
        serotypes[p['host_id']] = {'K': p['host_K'], 'O': p['host_O']}

    all_phages = sorted(interactions.keys())
    all_hosts = sorted(serotypes.keys())

    # ── Rank hosts given phage ──
    # Two metrics:
    #   `rank_hosts_ranks` — per (phage, positive host) pair (legacy).
    #   `phage_anyhit_ranks` — per phage with positives, the BEST rank
    #     among its positive hosts (PhageHostLearn-style "any-hit-in-top-k").
    rank_hosts_ranks = []
    phage_anyhit_ranks = []
    for phage in all_phages:
        pos_hosts = [h for h, label in interactions[phage].items() if label == 1]
        candidates = list(interactions[phage].keys())
        if not pos_hosts:
            continue

        proteins = phage_protein_map.get(phage, set())
        ranked = rank_hosts(predictor, proteins, candidates, serotypes,
                            emb_dict, pid_md5)
        if not ranked:
            continue

        host_to_rank = _ranks_with_ties(ranked, tie_method=tie_method)
        per_phage_ranks = []
        for pos_h in pos_hosts:
            if pos_h in host_to_rank:
                rank_hosts_ranks.append(host_to_rank[pos_h])
                per_phage_ranks.append(host_to_rank[pos_h])
        if per_phage_ranks:
            phage_anyhit_ranks.append(min(per_phage_ranks))

    # ── Rank phages given host ──
    # Symmetric: `host_anyhit_ranks` = per host with positives, best
    # rank among its positive phages.
    host_interactions = defaultdict(dict)
    for phage in interactions:
        for host, label in interactions[phage].items():
            host_interactions[host][phage] = label

    rank_phages_ranks = []
    host_anyhit_ranks = []
    for host in sorted(host_interactions.keys()):
        pos_phages = [ph for ph, label in host_interactions[host].items()
                      if label == 1]
        candidates = list(host_interactions[host].keys())
        if not pos_phages:
            continue

        ranked = rank_phages(predictor, host, candidates, phage_protein_map,
                             serotypes, emb_dict, pid_md5)
        if not ranked:
            continue

        phage_to_rank = _ranks_with_ties(ranked, tie_method=tie_method)
        per_host_ranks = []
        for pos_ph in pos_phages:
            if pos_ph in phage_to_rank:
                rank_phages_ranks.append(phage_to_rank[pos_ph])
                per_host_ranks.append(phage_to_rank[pos_ph])
        if per_host_ranks:
            host_anyhit_ranks.append(min(per_host_ranks))

    return {
        'rank_hosts': {
            # legacy per-pair counts
            'n_pairs': len(rank_hosts_ranks),
            'hr_at_k': hr_curve(rank_hosts_ranks, max_k),       # per-pair (legacy)
            'mrr': mrr(rank_hosts_ranks),
            # new: phage-level any-hit (PHL-style headline)
            'n_phages_with_pos': len(phage_anyhit_ranks),
            'hr_at_k_any_hit': hr_curve(phage_anyhit_ranks, max_k),
            'mrr_any_hit': mrr(phage_anyhit_ranks),
            'n_phages': len(all_phages),
            'n_hosts': len(all_hosts),
        },
        'rank_phages': {
            'n_pairs': len(rank_phages_ranks),
            'hr_at_k': hr_curve(rank_phages_ranks, max_k),
            'mrr': mrr(rank_phages_ranks),
            'n_hosts_with_pos': len(host_anyhit_ranks),
            'hr_at_k_any_hit': hr_curve(host_anyhit_ranks, max_k),
            'mrr_any_hit': mrr(host_anyhit_ranks),
            'n_phages': len(all_phages),
            'n_hosts': len(all_hosts),
        },
    }
