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
    prot_embs = [emb_dict[m] for m in prot_md5s if m in emb_dict]

    if not prot_embs:
        return []

    host_scores = []
    for host in candidate_hosts:
        if host not in serotypes:
            continue
        host_k = serotypes[host]['K']
        host_o = serotypes[host]['O']
        score = predictor.score_pair(prot_embs, host_k, host_o)
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
        prot_embs = [emb_dict[m] for m in prot_md5s if m in emb_dict]

        score = predictor.score_pair(prot_embs, host_k, host_o)
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

    Denominator semantics (changed 2026-04-27): every positive (phage, host)
    pair in the dataset contributes to ``n_pairs`` and to HR@k. If the model
    cannot score a positive pair — for example because the host's K-type
    falls outside the model's training vocabulary, or the host has
    null serotype values, or the phage has no scorable proteins — the
    positive is counted as a miss (rank = ∞), not silently dropped.

    This was previously a real methodological bug: models trained on
    restricted class vocabularies (e.g. v1 highconf with 64 of 161
    K-classes) could quote inflated HR@1 numbers because their denominator
    excluded all the pairs they couldn't score. With this fix, any pair the
    model fails to score counts against its hit rate, so HR@k numbers from
    different models are directly comparable on a fixed denominator (the
    full validation set).

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
            n_pairs       - total number of positive pairs (fixed denominator)
            n_pairs_scored - number of those pairs the model actually scored
            n_phages
            n_hosts
            hr_at_k
            mrr
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

    UNSCORABLE = float('inf')

    # ── Rank hosts given phage ──
    rank_hosts_ranks = []
    n_hosts_scored = 0
    for phage in all_phages:
        pos_hosts = [h for h, label in interactions[phage].items() if label == 1]
        if not pos_hosts:
            continue
        candidates = list(interactions[phage].keys())

        proteins = phage_protein_map.get(phage, set())
        ranked = rank_hosts(predictor, proteins, candidates, serotypes,
                            emb_dict, pid_md5)
        host_to_rank = _ranks_with_ties(ranked, tie_method=tie_method) if ranked else {}

        for pos_h in pos_hosts:
            if pos_h in host_to_rank:
                rank_hosts_ranks.append(host_to_rank[pos_h])
                n_hosts_scored += 1
            else:
                # Positive host could not be scored — counts as a miss.
                # rank = ∞ fails HR@k for any finite k and contributes 0 to MRR.
                rank_hosts_ranks.append(UNSCORABLE)

    # ── Rank phages given host ──
    host_interactions = defaultdict(dict)
    for phage in interactions:
        for host, label in interactions[phage].items():
            host_interactions[host][phage] = label

    rank_phages_ranks = []
    n_phages_scored = 0
    for host in sorted(host_interactions.keys()):
        pos_phages = [ph for ph, label in host_interactions[host].items()
                      if label == 1]
        if not pos_phages:
            continue
        candidates = list(host_interactions[host].keys())

        ranked = rank_phages(predictor, host, candidates, phage_protein_map,
                             serotypes, emb_dict, pid_md5)
        phage_to_rank = _ranks_with_ties(ranked, tie_method=tie_method) if ranked else {}

        for pos_ph in pos_phages:
            if pos_ph in phage_to_rank:
                rank_phages_ranks.append(phage_to_rank[pos_ph])
                n_phages_scored += 1
            else:
                rank_phages_ranks.append(UNSCORABLE)

    return {
        'rank_hosts': {
            'n_pairs': len(rank_hosts_ranks),
            'n_pairs_scored': n_hosts_scored,
            'n_phages': len(all_phages),
            'n_hosts': len(all_hosts),
            'hr_at_k': hr_curve(rank_hosts_ranks, max_k),
            'mrr': mrr(rank_hosts_ranks),
        },
        'rank_phages': {
            'n_pairs': len(rank_phages_ranks),
            'n_pairs_scored': n_phages_scored,
            'n_phages': len(all_phages),
            'n_hosts': len(all_hosts),
            'hr_at_k': hr_curve(rank_phages_ranks, max_k),
            'mrr': mrr(rank_phages_ranks),
        },
    }
