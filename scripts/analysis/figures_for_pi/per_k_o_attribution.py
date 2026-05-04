"""Per-K-type and per-O-type attribution of the 6-source OR ensemble wins.

For each PHL phage hit by the ensemble at rank 1, identify which
sources contributed and which was the sole winner. Bucket the
attribution by the phage's true K-set (any K in the phage's positive
host set) and true O-set (any O in the phage's positive host set).

Outputs:
  - results/figures/knob_comparisons/attribution_by_K.csv
  - results/figures/knob_comparisons/attribution_by_O.csv
  - results/figures/knob_comparisons/attribution_summary.txt
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PER_PHAGE_CSV = REPO / "results/figures/knob_comparisons/three_family_or_ensemble_per_phage.csv"
INTERACTION = REPO / "data/validation_data/HOST_RANGE/PhageHostLearn/metadata/interaction_matrix.tsv"
OUT_DIR = REPO / "results/figures/knob_comparisons"

SOURCES = ["ESM-2_K", "ESM-2_O", "ProtT5_K", "ProtT5_O", "kmer_K", "kmer_O"]


def normalise_label(s):
    """Strip whitespace; treat empty / null as None."""
    s = (s or "").strip()
    return s if s and s.lower() not in ("null", "nan", "none", "") else None


def load_phl_truth():
    """Return {phage_id: (set of true K, set of true O)} for PHL positives only."""
    K = defaultdict(set)
    O = defaultdict(set)
    with open(INTERACTION) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("label") != "1":
                continue
            phage = r["phage_id"]
            k = normalise_label(r.get("host_K"))
            o = normalise_label(r.get("host_O"))
            if k:
                K[phage].add(k)
            if o:
                O[phage].add(o)
    return K, O


def load_per_phage_ranks():
    """Return {phage: {source: rank}} for PHL phages only."""
    out = {}
    with open(PER_PHAGE_CSV) as f:
        for r in csv.DictReader(f):
            if r["dataset"] != "PhageHostLearn":
                continue
            phage = r["phage_id"]
            ranks = {s: int(r[s]) for s in SOURCES}
            out[phage] = ranks
    return out


def main():
    K_truth, O_truth = load_phl_truth()
    ranks = load_per_phage_ranks()

    # Identify ensemble wins (any source at rank ≤ 1)
    wins = {p: r for p, r in ranks.items() if min(r.values()) <= 1}
    print(f"PHL phages: {len(ranks)}  ensemble-hit at rank 1: {len(wins)}")

    # ---- Per-K attribution ----
    # For each K-type, count phages whose true-K-set contains it,
    # and per source: contributed (rank ≤ 1), was sole winner
    K_phages = defaultdict(set)            # K-type -> set of phages owning it
    K_winning_phages = defaultdict(set)    # K-type -> set of phages where ensemble won
    K_contributed = defaultdict(lambda: defaultdict(int))  # K -> source -> count
    K_sole = defaultdict(lambda: defaultdict(int))         # K -> source -> count

    for phage, ks in K_truth.items():
        for k in ks:
            K_phages[k].add(phage)
            if phage in wins:
                K_winning_phages[k].add(phage)
                winners = [s for s, r in wins[phage].items() if r <= 1]
                for s in winners:
                    K_contributed[k][s] += 1
                if len(winners) == 1:
                    K_sole[k][winners[0]] += 1

    # Order K-types by number of phages that own it
    k_order = sorted(K_phages, key=lambda k: -len(K_phages[k]))

    out_csv = OUT_DIR / "attribution_by_K.csv"
    with open(out_csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["K_type", "n_phages_with_this_K", "n_ensemble_wins"]
                   + [f"contrib_{s}" for s in SOURCES]
                   + [f"sole_{s}" for s in SOURCES])
        for k in k_order:
            n_total = len(K_phages[k])
            n_win = len(K_winning_phages[k])
            row = [k, n_total, n_win] \
                  + [K_contributed[k][s] for s in SOURCES] \
                  + [K_sole[k][s] for s in SOURCES]
            w.writerow(row)
    print(f"  wrote {out_csv}")

    # ---- Per-O attribution ----
    O_phages = defaultdict(set)
    O_winning_phages = defaultdict(set)
    O_contributed = defaultdict(lambda: defaultdict(int))
    O_sole = defaultdict(lambda: defaultdict(int))

    for phage, os_ in O_truth.items():
        for o in os_:
            O_phages[o].add(phage)
            if phage in wins:
                O_winning_phages[o].add(phage)
                winners = [s for s, r in wins[phage].items() if r <= 1]
                for s in winners:
                    O_contributed[o][s] += 1
                if len(winners) == 1:
                    O_sole[o][winners[0]] += 1

    o_order = sorted(O_phages, key=lambda o: -len(O_phages[o]))

    out_csv2 = OUT_DIR / "attribution_by_O.csv"
    with open(out_csv2, "w") as f:
        w = csv.writer(f)
        w.writerow(["O_type", "n_phages_with_this_O", "n_ensemble_wins"]
                   + [f"contrib_{s}" for s in SOURCES]
                   + [f"sole_{s}" for s in SOURCES])
        for o in o_order:
            n_total = len(O_phages[o])
            n_win = len(O_winning_phages[o])
            row = [o, n_total, n_win] \
                  + [O_contributed[o][s] for s in SOURCES] \
                  + [O_sole[o][s] for s in SOURCES]
            w.writerow(row)
    print(f"  wrote {out_csv2}")

    # ---- Summary ----
    summary_lines = []
    summary_lines.append("=" * 70)
    summary_lines.append("Per-K and per-O attribution of 6-source OR ensemble PHL wins")
    summary_lines.append("=" * 70)
    summary_lines.append(f"\nTotal PHL phages: {len(ranks)}")
    summary_lines.append(f"Ensemble-hit at rank 1: {len(wins)}\n")

    # Sole-winner concentration: for each source's sole wins, which K-types do they cover?
    summary_lines.append("\n=== Sole-winner concentration per source (K-types) ===")
    summary_lines.append(f"  {'source':10s}  {'total sole wins':>15s}  K-type breakdown")
    src_to_k_sole = defaultdict(lambda: defaultdict(int))
    src_to_o_sole = defaultdict(lambda: defaultdict(int))
    for phage, r in wins.items():
        winners = [s for s, rk in r.items() if rk <= 1]
        if len(winners) != 1:
            continue
        src = winners[0]
        for k in K_truth.get(phage, []):
            src_to_k_sole[src][k] += 1
        for o in O_truth.get(phage, []):
            src_to_o_sole[src][o] += 1

    for s in SOURCES:
        ks = src_to_k_sole[s]
        total = sum(ks.values()) if ks else 0
        if total == 0:
            summary_lines.append(f"  {s:10s}  {'0':>15s}  —")
            continue
        # Note: a phage with multiple Ks contributes to each K count; total may exceed sole-win phages
        ks_str = ", ".join(f"{k}({n})" for k, n in sorted(ks.items(), key=lambda kv: -kv[1]))
        summary_lines.append(f"  {s:10s}  {total:>15d}  {ks_str}")

    summary_lines.append("\n=== Sole-winner concentration per source (O-types) ===")
    summary_lines.append(f"  {'source':10s}  {'total sole wins':>15s}  O-type breakdown")
    for s in SOURCES:
        os_ = src_to_o_sole[s]
        total = sum(os_.values()) if os_ else 0
        if total == 0:
            summary_lines.append(f"  {s:10s}  {'0':>15s}  —")
            continue
        os_str = ", ".join(f"{o}({n})" for o, n in sorted(os_.items(), key=lambda kv: -kv[1]))
        summary_lines.append(f"  {s:10s}  {total:>15d}  {os_str}")

    # Top-15 K-types: ensemble win rate + dominant contributor
    summary_lines.append("\n=== Top-15 K-types by phage count ===")
    summary_lines.append(f"  {'K':8s} {'n':>3s} {'win':>3s} {'rate':>5s}  {'top-contrib':12s}  {'sole-winners':12s}")
    for k in k_order[:15]:
        n = len(K_phages[k])
        w = len(K_winning_phages[k])
        rate = f"{w/n:.2f}"
        contribs = sorted(K_contributed[k].items(), key=lambda kv: -kv[1])
        top = contribs[0][0] if contribs and contribs[0][1] > 0 else "—"
        soles = sorted(K_sole[k].items(), key=lambda kv: -kv[1])
        sole_str = ",".join(f"{s}:{n}" for s, n in soles if n > 0) or "—"
        summary_lines.append(f"  {k:8s} {n:>3d} {w:>3d} {rate:>5s}  {top:12s}  {sole_str:30s}")

    summary_lines.append("\n=== Top-15 O-types by phage count ===")
    summary_lines.append(f"  {'O':12s} {'n':>3s} {'win':>3s} {'rate':>5s}  {'top-contrib':12s}  {'sole-winners':30s}")
    for o in o_order[:15]:
        n = len(O_phages[o])
        w = len(O_winning_phages[o])
        rate = f"{w/n:.2f}"
        contribs = sorted(O_contributed[o].items(), key=lambda kv: -kv[1])
        top = contribs[0][0] if contribs and contribs[0][1] > 0 else "—"
        soles = sorted(O_sole[o].items(), key=lambda kv: -kv[1])
        sole_str = ",".join(f"{s}:{n}" for s, n in soles if n > 0) or "—"
        summary_lines.append(f"  {o:12s} {n:>3d} {w:>3d} {rate:>5s}  {top:12s}  {sole_str:30s}")

    out_txt = OUT_DIR / "attribution_summary.txt"
    out_txt.write_text("\n".join(summary_lines) + "\n")
    print(f"  wrote {out_txt}")
    print()
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
