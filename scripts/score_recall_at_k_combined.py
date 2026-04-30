#!/usr/bin/env python3
"""
Score TropiSEQ, TropiGAT, and their UNION (TropiSEQ & TropiGAT combined)
at k = 1..K, following the authors' methodology in B.Compile_results.ipynb.

Authors' method (verbatim, summarised):
    For each phage P:
        For each model M:
            phage_pred_at_k(P, M) = ⋃_{protein p in P} top_k(p, M)
        combined_pred_at_k(P) = phage_pred_at_k(P, TropiGAT) ∪ phage_pred_at_k(P, TropiSEQ)
    Recall@k(M) = #{P : ground_truth_K(P) ∩ phage_pred_at_k(P, M) ≠ ∅} / #{P}

  - Per-phage prediction is UNION of per-protein top-k lists.
  - Combined model is UNION of TropiGAT and TropiSEQ phage-level predictions.
  - Recall is "any positive K of phage P found in phage_pred_at_k". A
    phage with multiple positive Ks scores 1 if any single one matches.

Inputs:
    --tropiseq-protein-tsv : per_protein_predictions.tsv from TropiSEQ inference
    --tropigat-protein-tsv : per_protein_predictions.tsv from TropiGAT inference
    --val-dir              : cipher's data/validation_data/HOST_RANGE/
    --out-dir              : output directory
    --max-k                : compute Recall@k for k = 1..max_k

Outputs:
    recall_at_k_combined.tsv  one row per (dataset, model) with R@1..R@K
    per_phage_diagnostics.tsv per-phage detail with hits at k=1
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tropiseq-protein-tsv", required=True)
    p.add_argument("--tropigat-protein-tsv", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-k", type=int, default=20)
    return p.parse_args()


def norm(s: str) -> str:
    if not s: return ""
    s = s.strip()
    if s.startswith("KL"):
        return "K" + s[2:]
    return s


def parse_top(s: str):
    if not s: return []
    return [norm(part.split(":")[0].strip()) for part in s.split(";") if ":" in part]


def load_per_protein(path):
    """Return dict: protein_id → ranked list of K-types (best first, normalized)."""
    out = {}
    with open(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        # column name varies; try common ones
        candidates = ["top_kl_predictions", "top_kl_max_pooled"]
        for row in r:
            for c in candidates:
                if c in row:
                    out[row["protein_id"]] = parse_top(row[c])
                    break
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = Path(args.val_dir)
    K = args.max_k

    ts_protein = load_per_protein(args.tropiseq_protein_tsv)
    tg_protein = load_per_protein(args.tropigat_protein_tsv)
    print(f"TropiSEQ proteins with predictions: {len(ts_protein)}")
    print(f"TropiGAT proteins with predictions: {len(tg_protein)}")

    # Per-dataset, per-phage scoring
    rows_recall = []  # (dataset, model, R@1, ..., R@K)
    rows_diag = []    # per-phage diagnostics (k=1)

    for dataset_dir in sorted(val_dir.iterdir()):
        ds = dataset_dir.name
        im = dataset_dir / "metadata" / "interaction_matrix.tsv"
        ppm = dataset_dir / "metadata" / "phage_protein_mapping.csv"
        if not im.exists() or not ppm.exists(): continue

        # phage → set of proteins
        phage_proteins = defaultdict(list)
        with open(ppm) as f:
            next(f)
            for line in f:
                ph, pr = line.strip().split(",")
                phage_proteins[ph].append(pr)

        # phage → set of positive K-types (any host where label=1).
        # Fixed-denominator policy (cipher project, 2026-04-30): every
        # phage with ≥1 positive record counts. Phages whose positives
        # all have null/empty host_K are KEPT in the denominator with
        # an empty positive-K set; they always miss (pos & pred = ∅).
        phages_with_pos = set()
        phage_pos_K = defaultdict(set)
        with open(im) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row.get("label") in ("1", 1, True):
                    ph = row["phage_id"]
                    phages_with_pos.add(ph)
                    k = norm(row.get("host_K", ""))
                    if k: phage_pos_K[ph].add(k)

        phages = sorted(phages_with_pos)
        n_phages = len(phages)

        # For each model, compute hit per phage at each k
        per_model_hits = {"TropiSEQ": [[0]*n_phages for _ in range(K+1)],
                          "TropiGAT": [[0]*n_phages for _ in range(K+1)],
                          "Combined": [[0]*n_phages for _ in range(K+1)]}

        for i, ph in enumerate(phages):
            pos = phage_pos_K[ph]
            proteins = phage_proteins.get(ph, [])
            # Build per-protein ranked lists for each model
            ts_lists = [ts_protein[p] for p in proteins if p in ts_protein]
            tg_lists = [tg_protein[p] for p in proteins if p in tg_protein]

            for k in range(1, K+1):
                ts_phage_pred = set().union(*[set(L[:k]) for L in ts_lists]) if ts_lists else set()
                tg_phage_pred = set().union(*[set(L[:k]) for L in tg_lists]) if tg_lists else set()
                comb_pred = ts_phage_pred | tg_phage_pred

                if pos & ts_phage_pred:    per_model_hits["TropiSEQ"][k][i] = 1
                if pos & tg_phage_pred:    per_model_hits["TropiGAT"][k][i] = 1
                if pos & comb_pred:        per_model_hits["Combined"][k][i] = 1

            # k=1 diagnostics
            ts_k1 = set().union(*[set(L[:1]) for L in ts_lists]) if ts_lists else set()
            tg_k1 = set().union(*[set(L[:1]) for L in tg_lists]) if tg_lists else set()
            rows_diag.append({
                "dataset": ds,
                "phage_id": ph,
                "n_proteins": len(proteins),
                "n_proteins_with_ts_pred": len(ts_lists),
                "n_proteins_with_tg_pred": len(tg_lists),
                "positive_K_types": ";".join(sorted(pos)),
                "ts_top1": ";".join(sorted(ts_k1)),
                "tg_top1": ";".join(sorted(tg_k1)),
                "ts_hit@1": int(bool(pos & ts_k1)),
                "tg_hit@1": int(bool(pos & tg_k1)),
                "combined_hit@1": int(bool(pos & (ts_k1 | tg_k1))),
            })

        for model in ["TropiSEQ", "TropiGAT", "Combined"]:
            recalls = [sum(per_model_hits[model][k])/n_phages if n_phages else 0.0
                       for k in range(1, K+1)]
            rows_recall.append({"dataset": ds, "model": model, "n_phages": n_phages,
                                **{f"R@{k}": recalls[k-1] for k in range(1, K+1)}})

    # Write outputs
    cols = ["dataset", "model", "n_phages"] + [f"R@{k}" for k in range(1, K+1)]
    with open(out_dir / "recall_at_k_combined.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows_recall:
            w.writerow(r)

    diag_cols = ["dataset", "phage_id", "n_proteins", "n_proteins_with_ts_pred",
                 "n_proteins_with_tg_pred", "positive_K_types",
                 "ts_top1", "tg_top1", "ts_hit@1", "tg_hit@1", "combined_hit@1"]
    with open(out_dir / "per_phage_diagnostics.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=diag_cols, delimiter="\t")
        w.writeheader()
        for r in rows_diag:
            w.writerow(r)

    # Console summary
    print(f"\n=== Recall@k (k=1..{K}) — phage-level any-hit, strict denominator ===")
    by_ds = defaultdict(dict)
    for r in rows_recall:
        by_ds[r["dataset"]][r["model"]] = r
    for ds in sorted(by_ds.keys()):
        n = by_ds[ds]["TropiSEQ"]["n_phages"]
        print(f"\n{ds}  (n={n})")
        header = f"  {'model':<10} | " + " ".join(f"R@{k:>2}" for k in [1,2,3,5,10,15,20])
        print(header)
        print("  " + "-" * (len(header) - 2))
        for model in ["TropiSEQ", "TropiGAT", "Combined"]:
            r = by_ds[ds][model]
            vals = " ".join(f"{r[f'R@{k}']*100:>4.1f}" for k in [1,2,3,5,10,15,20])
            print(f"  {model:<10} | {vals}")


if __name__ == "__main__":
    main()
