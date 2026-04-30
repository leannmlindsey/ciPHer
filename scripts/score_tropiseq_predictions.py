#!/usr/bin/env python3
"""
Score TropiSEQ per-phage predictions against cipher's validation
interaction matrices.

Two metrics:
    Recall@N (phage-level, any-hit):
        For each phage in dataset's interaction matrix,
        K_pos(P) = {host_K : (P, host) is positive}
        Recall@N(P) = 1 if any K in K_pos(P) is in top-N predicted KL
                     by TropiSEQ for P, else 0
        Reported as mean over phages.
        Denominator includes phages where TropiSEQ has no prediction
        (those count as 0). This is the "strict" metric matching
        cipher's HR@k convention.

    Recall@N (pair-level):
        For each positive (phage, host) pair, check if host_K is in
        top-N predicted KL. Mean over pairs.

Outputs:
    recall_at_n_phage.tsv  — phage-level any-hit by dataset and N=1..10
    recall_at_n_pair.tsv   — pair-level by dataset and N=1..10
    per_phage_diagnostics.tsv — per-phage detail: K_pos, top1 pred, hit at N=1
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-phage-tsv", required=True,
                   help="per_phage_predictions.tsv (output of run_tropiseq_inference.py)")
    p.add_argument("--val-dir", required=True,
                   help="cipher data/validation_data/HOST_RANGE/")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    return p.parse_args()


def norm_k(s: str) -> str:
    """Normalize K-type string. TropiSEQ uses 'KL151', cipher uses 'K151'.
    Strip leading 'KL' or 'K' to a canonical 'K151' form."""
    if s is None: return ""
    s = s.strip()
    if s.startswith("KL"):
        return "K" + s[2:]
    return s


def parse_top_kl(s: str):
    """Parse 'KL151:0.599 ; KL37:0.235 ; ...' → ordered list of (norm_K, prob)."""
    if not s: return []
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part: continue
        kl, p = part.rsplit(":", 1)
        out.append((norm_k(kl.strip()), float(p)))
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = Path(args.val_dir)

    # Load per-phage TropiSEQ predictions
    preds = {}  # (dataset, phage) → list of (K, prob) sorted desc
    with open(args.per_phage_tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            preds[(row["dataset"], row["phage_id"])] = parse_top_kl(row["top_kl_max_pooled"])

    # For each dataset, load interaction matrix and score
    phage_rows = []
    pair_results = defaultdict(list)
    phage_results = defaultdict(list)
    for dataset_dir in sorted(val_dir.iterdir()):
        ds = dataset_dir.name
        im_path = dataset_dir / "metadata" / "interaction_matrix.tsv"
        if not im_path.exists():
            continue

        # Fixed-denominator policy (cipher project, 2026-04-30): every
        # phage with ≥1 positive record counts. Phages whose positives
        # all have null/empty host_K are KEPT with an empty positive-K
        # set; they always miss (pos & topN = ∅).
        phages_with_pos = set()
        phage_pos_K = defaultdict(set)
        positive_pairs = []
        with open(im_path) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("label") not in ("1", 1, True): continue
                ph = row["phage_id"]
                phages_with_pos.add(ph)
                k = norm_k(row.get("host_K", ""))
                if not k: continue
                phage_pos_K[ph].add(k)
                positive_pairs.append((ph, k))

        # Score each phage
        for ph in sorted(phages_with_pos):
            pos = phage_pos_K[ph]
            top = preds.get((ds, ph), [])
            top_k_ranked = [k for k, _ in top]

            for N in args.ks:
                topN = set(top_k_ranked[:N])
                hit = 1 if (pos & topN) else 0
                phage_results[(ds, N)].append(hit)
            top1_pred = top_k_ranked[0] if top_k_ranked else "NO_PRED"
            phage_rows.append({
                "dataset": ds,
                "phage_id": ph,
                "n_positive_hosts": len(pos),
                "positive_K_types": ";".join(sorted(pos)),
                "tropiseq_top1": top1_pred,
                "tropiseq_top5": ";".join(top_k_ranked[:5]),
                "hit_at_1": int(top1_pred in pos),
            })

        # Pair-level
        for (ph, target_k) in positive_pairs:
            top = preds.get((ds, ph), [])
            top_k_ranked = [k for k, _ in top]
            for N in args.ks:
                hit = 1 if target_k in set(top_k_ranked[:N]) else 0
                pair_results[(ds, N)].append(hit)

    # Write outputs
    with open(out_dir / "recall_at_n_phage.tsv", "w") as f:
        f.write("dataset\tn_phages\t" + "\t".join(f"recall@{N}" for N in args.ks) + "\n")
        datasets = sorted(set(d for (d, _) in phage_results.keys()))
        for ds in datasets:
            n = len(phage_results[(ds, args.ks[0])])
            recalls = [sum(phage_results[(ds, N)]) / n if n else 0.0 for N in args.ks]
            f.write(f"{ds}\t{n}\t" + "\t".join(f"{r:.4f}" for r in recalls) + "\n")

    with open(out_dir / "recall_at_n_pair.tsv", "w") as f:
        f.write("dataset\tn_positive_pairs\t" + "\t".join(f"recall@{N}" for N in args.ks) + "\n")
        datasets = sorted(set(d for (d, _) in pair_results.keys()))
        for ds in datasets:
            n = len(pair_results[(ds, args.ks[0])])
            recalls = [sum(pair_results[(ds, N)]) / n if n else 0.0 for N in args.ks]
            f.write(f"{ds}\t{n}\t" + "\t".join(f"{r:.4f}" for r in recalls) + "\n")

    with open(out_dir / "per_phage_diagnostics.tsv", "w") as f:
        cols = ["dataset", "phage_id", "n_positive_hosts", "positive_K_types",
                "tropiseq_top1", "tropiseq_top5", "hit_at_1"]
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in phage_rows:
            w.writerow(r)

    # Print summary
    print(f"\n=== Phage-level any-hit Recall@N (strict denominator) ===")
    print(f"{'dataset':<16} {'n_phages':>9} | " + " ".join(f"R@{N:<2}" for N in args.ks))
    for ds in sorted(set(d for (d, _) in phage_results.keys())):
        n = len(phage_results[(ds, args.ks[0])])
        recalls = [sum(phage_results[(ds, N)]) / n if n else 0.0 for N in args.ks]
        print(f"{ds:<16} {n:>9} | " + " ".join(f"{r*100:>4.1f}" for r in recalls))

    print(f"\n=== Pair-level Recall@N (strict denominator) ===")
    print(f"{'dataset':<16} {'n_pairs':>9} | " + " ".join(f"R@{N:<2}" for N in args.ks))
    for ds in sorted(set(d for (d, _) in pair_results.keys())):
        n = len(pair_results[(ds, args.ks[0])])
        recalls = [sum(pair_results[(ds, N)]) / n if n else 0.0 for N in args.ks]
        print(f"{ds:<16} {n:>9} | " + " ".join(f"{r*100:>4.1f}" for r in recalls))


if __name__ == "__main__":
    main()
