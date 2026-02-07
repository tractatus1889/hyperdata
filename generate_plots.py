#!/usr/bin/env python3
"""Generate plots for the hyperdata experiment report."""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
})

x_labels = ["just\nexamples", "1%", "5%", "10%"]
VARIANTS = ["examples", "hyperdata_1pct", "hyperdata_5pct", "hyperdata_10pct"]


def load_comparison(grammar):
    """Load comparison report for a grammar."""
    path = Path(f"results/{grammar}_comparison_report.json")
    with open(path) as f:
        return json.load(f)


def get_metrics(report):
    """Extract ordered metrics from a comparison report."""
    grammar = report["grammar"]
    validity = []
    ppl_ratio = []
    completion = []
    for variant in VARIANTS:
        key = f"pythia-1.4b_{grammar}_{variant}"
        m = report["models"][key]
        validity.append(m["generation_validity"] * 100)
        ppl_ratio.append(m["perplexity_ratio"])
        completion.append(m["completion_accuracy"] * 100)
    return validity, ppl_ratio, completion


def plot_grammar_overview(grammar, title):
    """Generate overview plot (generation validity + perplexity ratio) for a grammar."""
    report = load_comparison(grammar)
    validity, ppl_ratio, completion = get_metrics(report)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Generation validity
    colors = ["#6c757d", "#adb5bd", "#2196F3", "#64b5f6"]
    bars1 = ax1.bar(range(4), validity, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel("Validity Rate (%)")
    ax1.set_xlabel("Hyperdata Explanation Ratio")
    ax1.set_title("Generation Validity")

    # Auto-scale y axis
    min_val = min(validity)
    max_val = max(validity)
    margin = max((max_val - min_val) * 0.3, 1.0)
    ax1.set_ylim(max(0, min_val - margin), min(100, max_val + margin))
    ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))

    for bar, val in zip(bars1, validity):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + margin * 0.05,
                 f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

    # Right: Perplexity ratio
    # Color by which has highest PPL ratio
    best_idx = np.argmax(ppl_ratio)
    ppl_colors = ["#6c757d" if i == 0 else "#adb5bd" for i in range(4)]
    ppl_colors[best_idx] = "#2196F3"
    bars2 = ax2.bar(range(4), ppl_ratio, color=ppl_colors, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(x_labels)
    ax2.set_ylabel("Perplexity Ratio (invalid / valid)")
    ax2.set_xlabel("Hyperdata Explanation Ratio")
    ax2.set_title("Perplexity Discrimination")

    min_ppl = min(ppl_ratio)
    max_ppl = max(ppl_ratio)
    ppl_margin = max((max_ppl - min_ppl) * 0.3, 0.1)
    ax2.set_ylim(min_ppl - ppl_margin, max_ppl + ppl_margin)

    for bar, val in zip(bars2, ppl_ratio):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ppl_margin * 0.05,
                 f"{val:.2f}x", ha="center", va="bottom", fontweight="bold", fontsize=10)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"figures/{grammar}_overview.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_cross_grammar_comparison():
    """Generate cross-grammar comparison plot."""
    grammars = ["grammar1", "grammar2", "grammar3"]
    grammar_labels = ["Grammar 1\n(START/MID/END)", "Grammar 2\n(Color-Shape)", "Grammar 3\n(Palindrome)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for idx, (grammar, label) in enumerate(zip(grammars, grammar_labels)):
        report = load_comparison(grammar)
        validity, ppl_ratio, completion = get_metrics(report)

        ax = axes[idx]
        colors = ["#6c757d", "#adb5bd", "#2196F3", "#64b5f6"]
        bars = ax.bar(range(4), validity, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(4))
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(label, fontsize=11)

        if idx == 0:
            ax.set_ylabel("Generation Validity (%)")

        # Auto-scale
        min_val = min(validity)
        max_val = max(validity)
        margin = max((max_val - min_val) * 0.3, 1.0)
        ax.set_ylim(max(0, min_val - margin), min(100, max_val + margin))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))

        for bar, val in zip(bars, validity):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + margin * 0.05,
                    f"{val:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=9)

    fig.suptitle("Generation Validity Across Grammars", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/cross_grammar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# Generate all plots
for grammar, title in [
    ("grammar1", "Grammar 1: START/MID/END"),
    ("grammar2", "Grammar 2: Color-Shape Agreement"),
    ("grammar3", "Grammar 3: Palindromic Brackets"),
]:
    plot_grammar_overview(grammar, title)

plot_cross_grammar_comparison()

print("Saved figures to figures/")
