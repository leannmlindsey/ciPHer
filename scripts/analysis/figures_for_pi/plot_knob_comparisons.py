"""Knob-comparison figures for the PI meeting deck.

Reads results/experiment_log.csv and produces three grouped bar charts:
  1. fig_model_comparison.svg/png       — best HR@1 per (architecture × dataset)
  2. fig_embedding_comparison.svg/png   — best HR@1 per (embedding family × dataset)
  3. fig_filter_comparison.svg/png      — best HR@1 per (training filter × dataset)

Metric used: <DS>_best_anyhit_HR1 (any-hit, fixed strict denominator,
team-best across K-only / O-only / merged / OR per dataset). This is
the metric standardized in the 2026-04-27 broadcasts.

Each panel shows the structural ceiling as a thin grey dotted line per
the 2026-04-28 ceiling finding.

Run:
    python3 scripts/analysis/figures_for_pi/plot_knob_comparisons.py

Outputs land under results/figures/knob_comparisons/.
"""
from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[3]
CSV_PATH = REPO / "results" / "experiment_log.csv"
OUT_DIR = REPO / "results" / "figures" / "knob_comparisons"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["PhageHostLearn", "PBIP", "UCSD", "CHEN", "GORODNICHIV"]
DATASET_LABELS = {"PhageHostLearn": "PHL", "PBIP": "PBIP", "UCSD": "UCSD",
                  "CHEN": "CHEN", "GORODNICHIV": "GOR"}

# Structural ceilings from 2026-04-28_phl_structural_ceiling_missing_rbps.md.
# Using merged ceilings since best_anyhit_HR1 takes max over heads incl. OR.
CEILINGS = {
    "PhageHostLearn": 0.724,
    "PBIP": 0.971,
    "UCSD": 0.818,
    "CHEN": 1.000,
    "GORODNICHIV": 1.000,
}


def to_float(s):
    try:
        v = float(s)
        return v if not np.isnan(v) else None
    except (TypeError, ValueError):
        return None


def parse_filter(row) -> str:
    """Map a row to one of seven canonical filter buckets."""
    name = row.get("run_name", "")
    cli = row.get("cli_argv", "") or ""
    # Run-name shortcuts
    if "v3_uat" in name or "v3uat" in name:
        return "highconf_v3_uat"
    if "v3_strict" in name or "v3strict" in name:
        return "highconf_v3_strict"
    if "v2_uat" in name:
        return "highconf_v2_uat"
    if "v2_strict" in name:
        return "highconf_v2_strict"
    if "highconf" in name or "highcf" in name or "lab_prott5_xl_full_highconf" in name:
        return "highconf_v1"
    # CLI-arg fallbacks
    if "highconf_v3_multitop" in cli or "v3_multitop" in cli:
        return "highconf_v3_uat" if "uat" in cli.lower() else "highconf_v3_strict"
    if "highconf_v2" in cli:
        return "highconf_v2_uat" if "UAT" in cli else "highconf_v2_strict"
    if "highconf_pipeline_positive_K" in cli or "highconf_tsp_K" in cli:
        return "highconf_v1"
    if "--positive_list" in cli and "pipeline_positive" in cli:
        return "pipeline_positive"
    if "--positive_list_k" in cli or "--positive_list_o" in cli:
        return "highconf_v2_strict"  # default v2 if per-head but no clearer signal
    if "--tools" in cli:
        return "tools"
    return "tools"  # legacy default


def normalize_embedding(emb: str) -> str:
    """Collapse embedding labels into compact buckets for the chart."""
    if not emb:
        return "unknown"
    e = emb.lower()
    if "esm2_150m" in e:
        return "ESM-2 150M mean"
    if "esm2_650m_full" in e:
        return "ESM-2 650M full"
    if "esm2_650m_seg4" in e:
        return "ESM-2 650M seg4"
    if "esm2_650m_seg8" in e:
        return "ESM-2 650M seg8"
    if "esm2_650m_seg16" in e:
        return "ESM-2 650M seg16"
    if "esm2_650m" in e:
        return "ESM-2 650M mean"
    if "esm2_3b" in e:
        return "ESM-2 3B mean"
    if "prott5_xl_full" in e:
        return "ProtT5 full"
    if "prott5_xl_seg" in e:
        m = re.search(r"seg(\d+)", e)
        return f"ProtT5 seg{m.group(1)}" if m else "ProtT5 segN"
    if "prott5" in e:
        return "ProtT5 mean"
    if "kmer" in e:
        m = re.search(r"kmer_([a-z0-9]+_k\d+)", e)
        return f"k-mer {m.group(1)}" if m else "k-mer"
    return emb


def model_label(m: str) -> str:
    return {
        "attention_mlp": "AttentionMLP",
        "light_attention": "LightAttention",
        "light_attention_binary": "LightAttention\nBinary",
        "contrastive_encoder": "Contrastive\nEncoder",
    }.get(m, m)


def filter_label(f: str) -> str:
    return {
        "tools": "tools",
        "pipeline_positive": "pipeline\npositive",
        "highconf_v1": "highconf\nv1",
        "highconf_v2_strict": "highconf\nv2 strict",
        "highconf_v2_uat": "highconf\nv2 UAT",
        "highconf_v3_strict": "highconf\nv3 strict",
        "highconf_v3_uat": "highconf\nv3 UAT",
    }.get(f, f)


def load_rows():
    with open(CSV_PATH) as f:
        rows = list(csv.DictReader(f))
    # Keep only rows with populated any-hit metrics
    populated = []
    for r in rows:
        phl = to_float(r.get("PhageHostLearn_best_anyhit_HR1", ""))
        pbip = to_float(r.get("PBIP_best_anyhit_HR1", ""))
        if phl is not None or pbip is not None:
            populated.append(r)
    return populated


def best_per_group(rows, group_fn):
    """Return {group: {ds: best_HR1}} taking max across runs in each group."""
    out = defaultdict(lambda: {ds: None for ds in DATASETS})
    for r in rows:
        g = group_fn(r)
        for ds in DATASETS:
            v = to_float(r.get(f"{ds}_best_anyhit_HR1", ""))
            if v is None:
                continue
            cur = out[g][ds]
            if cur is None or v > cur:
                out[g][ds] = v
    return out


def grouped_bar(per_group, group_order, title, fname, figsize=(11, 5)):
    """Draw grouped bars: x = groups, color = dataset."""
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    n_groups = len(group_order)
    n_ds = len(DATASETS)
    bar_w = 0.85 / n_ds
    x = np.arange(n_groups)
    colors = plt.cm.tab10(np.linspace(0, 1, n_ds))

    for i, ds in enumerate(DATASETS):
        vals = [per_group[g][ds] if per_group[g].get(ds) is not None else 0
                for g in group_order]
        # Mark missing as a hatched short bar so it's visible.
        missing = [per_group[g].get(ds) is None for g in group_order]
        offset = (i - n_ds / 2 + 0.5) * bar_w
        bars = ax.bar(x + offset, vals, bar_w,
                      label=DATASET_LABELS[ds], color=colors[i],
                      edgecolor="black", linewidth=0.4)
        for j, bar in enumerate(bars):
            if missing[j]:
                bar.set_hatch("///")
                bar.set_alpha(0.3)
        # Ceiling tick at top of each bar group for this dataset
        for j, g in enumerate(group_order):
            ceil = CEILINGS[ds]
            ax.plot([x[j] + offset - bar_w / 2, x[j] + offset + bar_w / 2],
                    [ceil, ceil], color="grey", linestyle=":", linewidth=0.6, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(group_order, fontsize=9)
    ax.set_ylabel("Best HR@1 (any-hit, fixed denominator)")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=5, fontsize=8, frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.set_axisbelow(True)

    fig.savefig(OUT_DIR / f"{fname}.svg", format="svg")
    fig.savefig(OUT_DIR / f"{fname}.png", format="png", dpi=180)
    plt.close(fig)
    print(f"  wrote {fname}.svg + .png")


def fig_model(rows):
    per = best_per_group(rows, lambda r: r["model"])
    # Order with most-populated architectures first
    counts = {m: sum(1 for r in rows if r["model"] == m) for m in per.keys()}
    order = sorted(per.keys(), key=lambda m: -counts[m])
    grouped_bar(
        {model_label(m): per[m] for m in order},
        [model_label(m) for m in order],
        "Best HR@1 per architecture (across all training filters and embeddings)",
        "fig_model_comparison",
        figsize=(8, 5),
    )
    print("\nModel comparison data:")
    print(f"  {'Architecture':30s} {'PHL':>6s} {'PBIP':>6s} {'UCSD':>6s} {'CHEN':>6s} {'GOR':>6s} {'n_runs':>6s}")
    for m in order:
        d = per[m]
        n = counts[m]
        def fmt(x): return f"{x:.3f}" if x is not None else "  -  "
        print(f"  {model_label(m).replace(chr(10),' '):30s} "
              f"{fmt(d['PhageHostLearn']):>6s} {fmt(d['PBIP']):>6s} "
              f"{fmt(d['UCSD']):>6s} {fmt(d['CHEN']):>6s} {fmt(d['GORODNICHIV']):>6s} "
              f"{n:>6d}")


def fig_embedding(rows):
    per = best_per_group(rows, lambda r: normalize_embedding(r["embedding_type"]))
    # Custom ordering: ESM-2 by size+pooling, then ProtT5 by pooling, then k-mer
    embedding_order = [
        "ESM-2 150M mean",
        "ESM-2 650M mean",
        "ESM-2 650M seg4",
        "ESM-2 650M seg8",
        "ESM-2 650M seg16",
        "ESM-2 650M full",
        "ESM-2 3B mean",
        "ProtT5 mean",
        "ProtT5 seg4",
        "ProtT5 seg8",
        "ProtT5 seg16",
        "ProtT5 full",
        "k-mer aa20_k3",
        "k-mer aa20_k4",
        "k-mer murphy8_k5",
        "k-mer murphy10_k5",
        "k-mer li10_k5",
    ]
    order = [e for e in embedding_order if e in per]
    # Append any unrecognized embeddings at the end
    for e in per:
        if e not in order:
            order.append(e)
    grouped_bar(per, order,
                "Best HR@1 per embedding family / pooling (across all architectures and training filters)",
                "fig_embedding_comparison",
                figsize=(13, 5))
    print("\nEmbedding comparison data:")
    print(f"  {'Embedding':22s} {'PHL':>6s} {'PBIP':>6s} {'UCSD':>6s} {'CHEN':>6s} {'GOR':>6s} {'n':>4s}")
    for e in order:
        d = per[e]
        n = sum(1 for r in rows if normalize_embedding(r["embedding_type"]) == e)
        def fmt(x): return f"{x:.3f}" if x is not None else "  -  "
        print(f"  {e:22s} {fmt(d['PhageHostLearn']):>6s} {fmt(d['PBIP']):>6s} "
              f"{fmt(d['UCSD']):>6s} {fmt(d['CHEN']):>6s} {fmt(d['GORODNICHIV']):>6s} {n:>4d}")


def fig_filter(rows):
    per = best_per_group(rows, parse_filter)
    filter_order = [
        "tools",
        "pipeline_positive",
        "highconf_v1",
        "highconf_v2_strict",
        "highconf_v2_uat",
        "highconf_v3_strict",
        "highconf_v3_uat",
    ]
    order = [f for f in filter_order if f in per]
    for f in per:
        if f not in order:
            order.append(f)
    pretty_per = {filter_label(f): per[f] for f in order}
    grouped_bar(pretty_per,
                [filter_label(f) for f in order],
                "Best HR@1 per training-data filter (across all architectures and embeddings)",
                "fig_filter_comparison",
                figsize=(11, 5))
    print("\nFilter comparison data:")
    print(f"  {'Filter':22s} {'PHL':>6s} {'PBIP':>6s} {'UCSD':>6s} {'CHEN':>6s} {'GOR':>6s} {'n':>4s}")
    for f in order:
        d = per[f]
        n = sum(1 for r in rows if parse_filter(r) == f)
        def fmt(x): return f"{x:.3f}" if x is not None else "  -  "
        print(f"  {f:22s} {fmt(d['PhageHostLearn']):>6s} {fmt(d['PBIP']):>6s} "
              f"{fmt(d['UCSD']):>6s} {fmt(d['CHEN']):>6s} {fmt(d['GORODNICHIV']):>6s} {n:>4d}")


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} runs with populated any-hit metrics "
          f"(of {sum(1 for _ in csv.DictReader(open(CSV_PATH)))} total in CSV).")
    print(f"Output dir: {OUT_DIR}")
    print()
    fig_model(rows)
    fig_embedding(rows)
    fig_filter(rows)
    print(f"\nDone. {len(list(OUT_DIR.glob('*.svg')))} SVGs in {OUT_DIR}.")


if __name__ == "__main__":
    main()
