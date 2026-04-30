#!/usr/bin/env python3
"""
Compute Recall@k (k=1..K) under the authors' set-union per-phage
aggregation, for FOUR methods on the same denominator:

    TropiSEQ          (Costa et al. 2024 — BLAST + cluster KL)
    TropiGAT          (Costa et al. 2024 — 92 KL-specific GATv2 ensemble)
    Combined          (TropiSEQ ∪ TropiGAT — authors' published combination)
    Cipher            (attention_mlp K-head per-protein top-k, then ∪ over phage's proteins)

This makes the comparison apples-to-apples: every method uses
    phage_pred_at_k(P) = ⋃_{protein} top_k(protein)
    hit(P) = (positive_K ∩ phage_pred_at_k(P)) ≠ ∅

Strict denominator (phages with no predictions count as 0).
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
    p.add_argument("--cipher-protein-tsv", required=True)
    p.add_argument("--val-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-k", type=int, default=20)
    return p.parse_args()


def norm(s):
    if not s: return ""
    s = s.strip()
    if s.startswith("KL"): return "K" + s[2:]
    return s


def parse_top(s):
    if not s: return []
    return [norm(part.split(":")[0].strip()) for part in s.split(";") if ":" in part]


def load_per_protein(path):
    out = {}
    with open(path) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            for c in ("top_kl_predictions", "top_kl_max_pooled"):
                if c in row and row[c] is not None:
                    out[row["protein_id"]] = parse_top(row[c])
                    break
            else:
                out[row["protein_id"]] = []
    return out


def main():
    args = parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    val_dir = Path(args.val_dir)
    K = args.max_k

    methods = {
        "TropiSEQ": load_per_protein(args.tropiseq_protein_tsv),
        "TropiGAT": load_per_protein(args.tropigat_protein_tsv),
        "Cipher":   load_per_protein(args.cipher_protein_tsv),
    }
    print("Per-protein predictions loaded:")
    for k, v in methods.items():
        print(f"  {k}: {len(v)} proteins with prediction")

    rows_recall = []
    rows_diag = []

    for dataset_dir in sorted(val_dir.iterdir()):
        ds = dataset_dir.name
        im = dataset_dir / "metadata" / "interaction_matrix.tsv"
        ppm = dataset_dir / "metadata" / "phage_protein_mapping.csv"
        if not im.exists() or not ppm.exists(): continue

        phage_proteins = defaultdict(list)
        with open(ppm) as f:
            next(f)
            for line in f:
                ph, pr = line.strip().split(",")
                phage_proteins[ph].append(pr)

        # Fixed-denominator policy (cipher project, 2026-04-30): every
        # phage with ≥1 positive (phage, host) record counts. Phages
        # whose positives all have empty/null host_K are KEPT in the
        # denominator with an empty positive-K set; they always miss
        # (pos & pred = ∅), which is the policy-correct outcome rather
        # than an exclusion. Previously this loop dropped null-K phages
        # via `phage_pos_K.keys()` as the denominator (PHL: dropped 7
        # of 127 → n=120; PBIP: dropped 0 of 104).
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

        # 4 methods: TropiSEQ, TropiGAT, Combined, Cipher
        per_method_hits = {m: [[0]*n_phages for _ in range(K+1)]
                           for m in ("TropiSEQ", "TropiGAT", "Combined", "Cipher")}

        for i, ph in enumerate(phages):
            pos = phage_pos_K[ph]
            proteins = phage_proteins.get(ph, [])
            ts_lists = [methods["TropiSEQ"][p] for p in proteins if p in methods["TropiSEQ"]]
            tg_lists = [methods["TropiGAT"][p] for p in proteins if p in methods["TropiGAT"]]
            cp_lists = [methods["Cipher"][p]   for p in proteins if p in methods["Cipher"]]

            for k in range(1, K+1):
                ts_pred = set().union(*[set(L[:k]) for L in ts_lists]) if ts_lists else set()
                tg_pred = set().union(*[set(L[:k]) for L in tg_lists]) if tg_lists else set()
                cp_pred = set().union(*[set(L[:k]) for L in cp_lists]) if cp_lists else set()
                comb_pred = ts_pred | tg_pred

                if pos & ts_pred:   per_method_hits["TropiSEQ"][k][i] = 1
                if pos & tg_pred:   per_method_hits["TropiGAT"][k][i] = 1
                if pos & comb_pred: per_method_hits["Combined"][k][i] = 1
                if pos & cp_pred:   per_method_hits["Cipher"][k][i]   = 1

            ts_k1 = set().union(*[set(L[:1]) for L in ts_lists]) if ts_lists else set()
            tg_k1 = set().union(*[set(L[:1]) for L in tg_lists]) if tg_lists else set()
            cp_k1 = set().union(*[set(L[:1]) for L in cp_lists]) if cp_lists else set()
            rows_diag.append({
                "dataset": ds, "phage_id": ph,
                "n_proteins": len(proteins),
                "n_proteins_ts": len(ts_lists),
                "n_proteins_tg": len(tg_lists),
                "n_proteins_cp": len(cp_lists),
                "positive_K_types": ";".join(sorted(pos)),
                "ts_top1_set": ";".join(sorted(ts_k1)),
                "tg_top1_set": ";".join(sorted(tg_k1)),
                "cp_top1_set": ";".join(sorted(cp_k1)),
                "ts_hit@1": int(bool(pos & ts_k1)),
                "tg_hit@1": int(bool(pos & tg_k1)),
                "combined_hit@1": int(bool(pos & (ts_k1 | tg_k1))),
                "cp_hit@1": int(bool(pos & cp_k1)),
            })

        for m in ("TropiSEQ", "TropiGAT", "Combined", "Cipher"):
            recalls = [sum(per_method_hits[m][k])/n_phages if n_phages else 0.0
                       for k in range(1, K+1)]
            rows_recall.append({"dataset": ds, "model": m, "n_phages": n_phages,
                                **{f"R@{k}": recalls[k-1] for k in range(1, K+1)}})

    # Output
    cols = ["dataset", "model", "n_phages"] + [f"R@{k}" for k in range(1, K+1)]
    with open(out_dir / "recall_at_k_4way.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t")
        w.writeheader()
        for r in rows_recall: w.writerow(r)

    diag_cols = list(rows_diag[0].keys())
    with open(out_dir / "per_phage_diagnostics_4way.tsv", "w") as f:
        w = csv.DictWriter(f, fieldnames=diag_cols, delimiter="\t")
        w.writeheader()
        for r in rows_diag: w.writerow(r)

    # Console summary
    print(f"\n=== Recall@k (k=1..{K}), set-union aggregation, strict denominator ===")
    by_ds = defaultdict(dict)
    for r in rows_recall: by_ds[r["dataset"]][r["model"]] = r
    for ds in sorted(by_ds.keys()):
        n = by_ds[ds]["TropiSEQ"]["n_phages"]
        print(f"\n{ds}  (n={n} phages)")
        header = f"  {'model':<10} | " + " ".join(f"R@{k:>2}" for k in [1,2,3,5,10,15,20])
        print(header); print("  " + "-" * (len(header) - 2))
        for m in ("Cipher", "TropiSEQ", "TropiGAT", "Combined"):
            r = by_ds[ds][m]
            vals = " ".join(f"{r[f'R@{k}']*100:>4.1f}" for k in [1,2,3,5,10,15,20])
            print(f"  {m:<10} | {vals}")


if __name__ == "__main__":
    main()
