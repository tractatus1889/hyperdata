#!/usr/bin/env python3
"""Generate plots for the grammar1 report."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
from pathlib import Path

Path("figures").mkdir(exist_ok=True)

x = [0, 1, 5, 10]
x_labels = ["just\nexamples", "1%", "5%", "10%"]

# Style
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 12,
})

# --- Plot 1: Generation Validity ---
validity_rate = [97.6, 94.9, 99.2, 98.3]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(range(len(x)), validity_rate, color=["#6c757d", "#adb5bd", "#2196F3", "#64b5f6"],
              edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(x)))
ax.set_xticklabels(x_labels)
ax.set_ylabel("Validity Rate (%)")
ax.set_xlabel("Hyperdata Explanation Ratio")
ax.set_title("Generation Validity (n=6,000 per dataset)")
ax.set_ylim(93, 100)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))

for bar, val in zip(bars, validity_rate):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
            f"{val}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

plt.tight_layout()
plt.savefig("figures/grammar1_generation_validity.png", dpi=150)
plt.close()

# --- Plot 2: Perplexity Ratio ---
ppl_ratio = [16.46, 16.27, 16.30, 17.37]

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(range(len(x)), ppl_ratio, color=["#6c757d", "#adb5bd", "#64b5f6", "#2196F3"],
              edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(x)))
ax.set_xticklabels(x_labels)
ax.set_ylabel("Perplexity Ratio (invalid / valid)")
ax.set_xlabel("Hyperdata Explanation Ratio")
ax.set_title("Perplexity Discrimination")
ax.set_ylim(15.5, 18)

for bar, val in zip(bars, ppl_ratio):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            f"{val:.2f}x", ha="center", va="bottom", fontweight="bold", fontsize=11)

plt.tight_layout()
plt.savefig("figures/grammar1_perplexity_ratio.png", dpi=150)
plt.close()

# --- Plot 3: Combined overview (two subplots) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: Generation validity
bars1 = ax1.bar(range(len(x)), validity_rate, color=["#6c757d", "#adb5bd", "#2196F3", "#64b5f6"],
                edgecolor="black", linewidth=0.5)
ax1.set_xticks(range(len(x)))
ax1.set_xticklabels(x_labels)
ax1.set_ylabel("Validity Rate (%)")
ax1.set_xlabel("Hyperdata Explanation Ratio")
ax1.set_title("Generation Validity")
ax1.set_ylim(93, 100)
ax1.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f%%'))
for bar, val in zip(bars1, validity_rate):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
             f"{val}%", ha="center", va="bottom", fontweight="bold", fontsize=10)

# Right: Perplexity ratio
bars2 = ax2.bar(range(len(x)), ppl_ratio, color=["#6c757d", "#adb5bd", "#64b5f6", "#2196F3"],
                edgecolor="black", linewidth=0.5)
ax2.set_xticks(range(len(x)))
ax2.set_xticklabels(x_labels)
ax2.set_ylabel("Perplexity Ratio (invalid / valid)")
ax2.set_xlabel("Hyperdata Explanation Ratio")
ax2.set_title("Perplexity Discrimination")
ax2.set_ylim(15.5, 18)
for bar, val in zip(bars2, ppl_ratio):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
             f"{val:.2f}x", ha="center", va="bottom", fontweight="bold", fontsize=10)

plt.tight_layout()
plt.savefig("figures/grammar1_overview.png", dpi=150)
plt.close()

print("Saved figures to figures/")
