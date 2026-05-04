"""6-source OR ensemble: best K-head per family + best O-head per family.

Three families (ESM-2, ProtT5, k-mer) each contribute a K-head and an
O-head. For each (dataset, phage), the ensemble rank is the min over
all 6 sources of the phage's best-positive-host rank. A phage is
ensemble-hit at rank ≤ k if any of the 6 sources got a positive host
at rank ≤ k.

This is the host-rank OR semantics (same as cipher's existing 2-model
hybrid), extended to 6 sources. Vocabulary differences across models
are handled naturally — a model that can't predict a class effectively
never ranks its host at 1, so the union picks up another model's
prediction.

Top-1 is the headline; Top-k for k=1..20 is computed for the curve.

Outputs land under results/figures/knob_comparisons/.

Run:
    python3 scripts/analysis/figures_for_pi/plot_three_family_or_ensemble.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
TSV_DIR = REPO / "results" / "analysis" / "per_phage"
OUT_DIR = REPO / "results" / "figures" / "knob_comparisons"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 6-source ensemble (5 distinct runs, kmer reused for both heads)
SOURCES = {
    "ESM-2_K":  ("sweep_posList_esm2_650m_seg4_cl70", "k"),
    "ESM-2_O":  ("sweep_posList_esm2_3b_mean_cl70",   "o"),
    "ProtT5_K": ("la_v3_uat_prott5_xl_seg8",          "k"),
    "ProtT5_O": ("sweep_prott5_mean_cl70",            "o"),
    "kmer_K":   ("sweep_kmer_aa20_k4",                "k"),
    "kmer_O":   ("sweep_kmer_aa20_k4",                "o"),
}

# Reference points (from agent 1's 2026-05-01 broadcast)
REFERENCE = {
    "Cipher single best (kmer_aa20_k4 OR, intra-model)": "sweep_kmer_aa20_k4",
    "Cipher 2-model hybrid (esm2_3b K + kmer O)":        ("sweep_posList_esm2_3b_mean_cl70", "sweep_kmer_aa20_k4"),
}

DATASETS = ["PhageHostLearn", "PBIP", "UCSD", "CHEN", "GORODNICHIV"]
DATASET_LABEL = {"PhageHostLearn": "PHL", "PBIP": "PBIP", "UCSD": "UCSD",
                 "CHEN": "CHEN", "GORODNICHIV": "GOR"}


def load_tsv(run_name: str):
    """Return {(dataset, phage_id): (k_rank, o_rank)} from a per-phage TSV."""
    path = TSV_DIR / f"per_phage_{run_name}.tsv"
    if not path.exists():
        raise FileNotFoundError(f"Missing per-phage TSV: {path}")
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ds = r["dataset"]
            phage = r["phage_id"]
            try:
                k = int(r["k_only_rank"]) if r["k_only_rank"] else 10**9
                o = int(r["o_only_rank"]) if r["o_only_rank"] else 10**9
            except ValueError:
                k, o = 10**9, 10**9
            out[(ds, phage)] = (k, o)
    return out


def compute_ensemble_rank(per_run, sources):
    """For each (dataset, phage), return ensemble rank = min over the chosen sources."""
    # Index per-source rank maps
    source_rank = {}
    for source_name, (run_name, head) in sources.items():
        if run_name not in per_run:
            per_run[run_name] = load_tsv(run_name)
        m = {}
        for key, (kr, or_) in per_run[run_name].items():
            m[key] = kr if head == "k" else or_
        source_rank[source_name] = m
    # Union of all (dataset, phage) keys across sources
    all_keys = set()
    for m in source_rank.values():
        all_keys.update(m.keys())
    ensemble = {}
    per_source_ranks = {}
    for key in all_keys:
        ranks = []
        per_source = {}
        for source_name in sources:
            r = source_rank[source_name].get(key, 10**9)
            ranks.append(r)
            per_source[source_name] = r
        ensemble[key] = min(ranks)
        per_source_ranks[key] = per_source
    return ensemble, per_source_ranks


def hr_at_k(rank_map, dataset, k_max=20):
    """Return list [HR@1, HR@2, ..., HR@k_max] for a single dataset."""
    ranks = [r for (ds, _), r in rank_map.items() if ds == dataset]
    n = len(ranks)
    if n == 0:
        return [0.0] * k_max, 0
    out = []
    for k in range(1, k_max + 1):
        hits = sum(1 for r in ranks if r <= k)
        out.append(hits / n)
    return out, n


def main():
    per_run = {}  # cache for load_tsv

    # Compute ensemble rank
    ensemble, per_source = compute_ensemble_rank(per_run, SOURCES)

    # Compute reference benchmarks under same per-phage TSVs
    # (1) Best single model OR — kmer_aa20_k4, intra-model OR = min(k_rank, o_rank)
    kmer_run = per_run["sweep_kmer_aa20_k4"]
    single_or = {key: min(kr, or_) for key, (kr, or_) in kmer_run.items()}

    # (2) 2-model hybrid: K from esm2_3b + O from kmer
    esm2_3b = per_run.setdefault("sweep_posList_esm2_3b_mean_cl70", load_tsv("sweep_posList_esm2_3b_mean_cl70"))
    hybrid2 = {}
    for key in set(kmer_run) | set(esm2_3b):
        kr = esm2_3b.get(key, (10**9, 10**9))[0]
        or_ = kmer_run.get(key, (10**9, 10**9))[1]
        hybrid2[key] = min(kr, or_)

    # Print headline table
    print(f"\n{'='*70}\n6-source OR ensemble — HR@1 by dataset (Top-1 headline)\n{'='*70}\n")
    print(f"{'dataset':14s} {'n':>4s}  {'kmer_OR':>9s}  {'2-hybrid':>9s}  {'6-source':>9s}  {'Δ vs hybrid':>11s}")
    headline = []
    for ds in DATASETS:
        c1, n1 = hr_at_k(single_or, ds, 1)
        c2, n2 = hr_at_k(hybrid2, ds, 1)
        c3, n3 = hr_at_k(ensemble, ds, 1)
        delta = c3[0] - c2[0]
        headline.append((ds, n3, c1[0], c2[0], c3[0]))
        print(f"  {ds:14s}  {n3:>3d}  {c1[0]:>9.3f}  {c2[0]:>9.3f}  {c3[0]:>9.3f}  {delta:>+11.3f}")

    # Top-k curves
    K_MAX = 20
    print(f"\n{'='*70}\n6-source OR ensemble — HR@k curves\n{'='*70}\n")
    curves = {ds: {"single": None, "hybrid": None, "ensemble": None, "n": 0} for ds in DATASETS}
    for ds in DATASETS:
        s, n = hr_at_k(single_or, ds, K_MAX); curves[ds]["single"] = s
        h, _ = hr_at_k(hybrid2, ds, K_MAX);   curves[ds]["hybrid"] = h
        e, _ = hr_at_k(ensemble, ds, K_MAX);  curves[ds]["ensemble"] = e
        curves[ds]["n"] = n

    # Print PHL and PBIP curves
    for ds in ("PhageHostLearn", "PBIP"):
        c = curves[ds]
        print(f"\n  {ds} (n={c['n']}):")
        print(f"    {'k':>3s}  {'kmer_OR':>9s}  {'2-hybrid':>9s}  {'6-source':>9s}")
        for k in (1, 2, 3, 5, 10, 20):
            print(f"    {k:>3d}  {c['single'][k-1]:>9.3f}  {c['hybrid'][k-1]:>9.3f}  {c['ensemble'][k-1]:>9.3f}")

    # Per-source attribution at PHL HR@1 — which sources drove the ensemble?
    print(f"\n{'='*70}\nPHL Top-1 attribution: which sources drive each ensemble win?\n{'='*70}\n")
    phl_keys = [k for k in ensemble if k[0] == "PhageHostLearn"]
    win = sum(1 for k in phl_keys if ensemble[k] <= 1)
    contrib = defaultdict(int)
    sole_contrib = defaultdict(int)
    for key in phl_keys:
        if ensemble[key] > 1:
            continue
        winners = [src for src, r in per_source[key].items() if r <= 1]
        for src in winners:
            contrib[src] += 1
        if len(winners) == 1:
            sole_contrib[winners[0]] += 1
    print(f"  Total PHL phages hit at rank 1 by ensemble: {win}/{len(phl_keys)}")
    print(f"\n  Contribution count (a source contributes if its rank ≤ 1):")
    for src in SOURCES:
        c, sc = contrib[src], sole_contrib[src]
        print(f"    {src:10s}  contributed_in={c:>3d}  was_sole_winner_in={sc:>3d}")

    # === Plot — HR@k curves PHL + PBIP side by side ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, ds, title in [(axes[0], "PhageHostLearn", "PHL"), (axes[1], "PBIP", "PBIP")]:
        c = curves[ds]
        ks = list(range(1, K_MAX + 1))
        ax.plot(ks, c["single"], marker="o", label=f"Single (kmer_aa20_k4 OR)",
                color="#1c5fa3", linewidth=2)
        ax.plot(ks, c["hybrid"], marker="s", label=f"2-model hybrid (esm2_3b K + kmer O)",
                color="#2c8a3e", linewidth=2)
        ax.plot(ks, c["ensemble"], marker="^", label=f"6-source ensemble (3 families × 2 heads)",
                color="#b03030", linewidth=2.5)
        ax.set_title(f"{title} (n={c['n']})  Recall@k under host-rank OR")
        ax.set_xlabel("k")
        ax.set_ylabel("Recall@k (any-hit)")
        ax.set_ylim(0, 1.02)
        ax.set_xticks([1, 2, 3, 5, 10, 15, 20])
        ax.grid(linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", fontsize=9, frameon=True)

    fig.suptitle("3-Family OR Ensemble vs Single Model and 2-Model Hybrid",
                 fontsize=14, fontweight="bold", y=1.02)
    out_path_svg = OUT_DIR / "fig_three_family_or_ensemble.svg"
    out_path_png = OUT_DIR / "fig_three_family_or_ensemble.png"
    fig.savefig(out_path_svg, format="svg", bbox_inches="tight")
    fig.savefig(out_path_png, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {out_path_svg}")
    print(f"  wrote {out_path_png}")

    # === CSV outputs for downstream use ===
    out_csv = OUT_DIR / "three_family_or_ensemble_hrk.csv"
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n", "method"] + [f"HR@{k}" for k in range(1, K_MAX + 1)])
        for ds in DATASETS:
            c = curves[ds]
            for label, key in [("kmer_aa20_k4 single OR", "single"),
                               ("2-model hybrid (esm2_3b K + kmer O)", "hybrid"),
                               ("6-source ensemble", "ensemble")]:
                w.writerow([ds, c["n"], label] + [f"{x:.4f}" for x in c[key]])
    print(f"  wrote {out_csv}")

    # Per-phage detail dump
    out_per_phage = OUT_DIR / "three_family_or_ensemble_per_phage.csv"
    with open(out_per_phage, "w") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "phage_id", "ensemble_rank"] + list(SOURCES.keys()))
        for key in sorted(ensemble):
            ds, phage = key
            ranks = per_source[key]
            r_list = [ranks[s] for s in SOURCES]
            w.writerow([ds, phage, ensemble[key]] + r_list)
    print(f"  wrote {out_per_phage}")


if __name__ == "__main__":
    main()
