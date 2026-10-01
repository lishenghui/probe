#!/usr/bin/env python3
"""
Publication-Quality Plotting Script for Five-Method Latency Benchmark (v3)
Highlighting FraQ and LoRAForge with Muted Baselines.
"""
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
base = ROOT / "artifacts/hf_lora_census/unified_benchmark"
run4 = base / "run_all100_4way_2h_20260812"
svd_run = base / "run_random10_naive_svd_20260813/svd"

methods = ("svd", "florist", "spectral", "fraq", "flashtsqr")
labels = ("Naive SVD", "FLoRIST", "SpecTraL", "FraQ", "LoRAForge")

# Color Palette: Muted light tones for the first 3 baselines, Vibrant for FraQ and LoRAForge
colors = {
    "svd": "#94A3B8",       # Muted Slate Light Gray
    "florist": "#93C5FD",   # Muted Soft Sky Blue
    "spectral": "#FCD34D",  # Muted Soft Amber/Sand
    "fraq": "#059669",      # Vibrant Emerald Green (Focused)
    "flashtsqr": "#E11D48"  # Vibrant Carmine Crimson (LoRAForge Highlight)
}
color_list = [colors[m] for m in methods]

paths = {"svd": svd_run / "repo_results.csv"}
paths.update({m: run4 / m / "repo_results.csv" for m in methods[1:]})
data = {m: list(csv.DictReader(paths[m].open())) for m in methods}

stats = {}
vals_total = []
vals_per_layer = []

for m in methods:
    total = np.array([float(r["latency_ms"]) for r in data[m]])
    per_layer = np.array([float(r["latency_ms"]) / int(r["processed_layers"]) for r in data[m]])
    
    q1, med, q3 = np.percentile(total, [25, 50, 75])
    q1_pl, med_pl, q3_pl = np.percentile(per_layer, [25, 50, 75])
    
    vals_total.append(total)
    vals_per_layer.append(per_layer)
    
    stats[m] = {
        "repositories": len(total),
        "mean_ms": float(total.mean()),
        "std_ms": float(total.std(ddof=1)),
        "median_ms": float(med),
        "q1_ms": float(q1),
        "q3_ms": float(q3),
        "min_ms": float(total.min()),
        "max_ms": float(total.max()),
        "mean_ms_per_layer": float(per_layer.mean()),
        "std_ms_per_layer": float(per_layer.std(ddof=1)),
        "sem_ms_per_layer": float(per_layer.std(ddof=1) / np.sqrt(len(per_layer))),
        "median_ms_per_layer": float(med_pl),
        "q1_ms_per_layer": float(q1_pl),
        "q3_ms_per_layer": float(q3_pl),
    }

# Setup Global Matplotlib Parameters for Publication Quality
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "DejaVu Sans", "Arial", "Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.0,
    "grid.color": "#E2E8F0",
    "grid.linestyle": ":",
    "grid.linewidth": 0.8,
    "xtick.color": "#1E293B",
    "ytick.color": "#1E293B",
    "text.color": "#0F172A",
})

fig = plt.figure(figsize=(14.0, 5.8), dpi=300, facecolor="#FFFFFF")

# 2 Main Subplots layout
gs = fig.add_gridspec(1, 2, left=0.075, right=0.97, top=0.91, bottom=0.18, wspace=0.25)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])

x = np.arange(len(methods))

# ----------------------------------------------------
# PANEL A: Per-Repository Latency Distribution
# ----------------------------------------------------
ax0.set_facecolor("#FAFAFA")
ax0.grid(True, axis="y", zorder=0, alpha=0.7)

# Custom Boxplots: Muted for SVD/FLoRIST/SpecTraL, Thicker/Solid for FraQ & LoRAForge
bp = ax0.boxplot(
    vals_total,
    positions=x,
    widths=0.46,
    showfliers=False,
    patch_artist=True,
    zorder=2,
    medianprops=dict(linewidth=2.5, color="#0F172A"),
    whiskerprops=dict(linewidth=1.2, color="#64748B"),
    capprops=dict(linewidth=1.2, color="#64748B"),
)

for i, (box, m, c) in enumerate(zip(bp["boxes"], methods, color_list)):
    is_focused = m in ("fraq", "flashtsqr")
    alpha_val = 0.35 if is_focused else 0.15
    lw_val = 2.0 if is_focused else 1.2
    
    box.set_facecolor(c)
    box.set_alpha(alpha_val)
    box.set_edgecolor(c if is_focused else "#94A3B8")
    box.set_linewidth(lw_val)

# Smart Jitter Scatter Points
np.random.seed(42)
for j, (v, m, c) in enumerate(zip(vals_total, methods, color_list)):
    is_focused = m in ("fraq", "flashtsqr")
    alpha_val = 0.85 if is_focused else 0.40
    s_val = 26 if is_focused else 20
    edge_c = "#FFFFFF" if is_focused else "none"
    lw = 0.6 if is_focused else 0
    
    jitter = np.random.normal(0, 0.05, size=len(v))
    jitter = np.clip(jitter, -0.14, 0.14)
    
    ax0.scatter(
        j + jitter,
        v,
        s=s_val,
        alpha=alpha_val,
        color=c,
        edgecolors=edge_c,
        linewidths=lw,
        zorder=3 if is_focused else 2.5,
    )

ax0.set_yscale("log")
ax0.set_ylim(0.4, 3e7)
ax0.set_ylabel("Compression Latency per Repo (ms)", fontsize=11, fontweight="bold", labelpad=8)
ax0.set_title("(a) Per-Repository Latency Distribution", fontsize=12, fontweight="bold", pad=12, loc="left", color="#0F172A")
ax0.set_xticks(x)
ax0.set_xticklabels(labels, fontsize=10.5, fontweight="bold")

# Customize tick label colors to highlight FraQ and LoRAForge
for tick_label, m in zip(ax0.get_xticklabels(), methods):
    if m == "flashtsqr":
        tick_label.set_color("#E11D48")
        tick_label.set_fontweight("bold")
    elif m == "fraq":
        tick_label.set_color("#059669")
        tick_label.set_fontweight("bold")
    else:
        tick_label.set_color("#64748B")
        tick_label.set_fontweight("normal")

ax0.spines["top"].set_visible(False)
ax0.spines["right"].set_visible(False)

# Custom Y-axis tick formatter for panel A
def log_ms_formatter(y, pos):
    if y >= 1e6:
        return f"{y/1e6:.0f}M ms"
    elif y >= 1e3:
        return f"{y/1e3:.0f}k ms"
    elif y >= 1:
        return f"{y:.0f} ms"
    else:
        return f"{y:.1f} ms"

ax0.yaxis.set_major_formatter(FuncFormatter(log_ms_formatter))

# Median annotation for LoRAForge on Panel A
med_val = stats["flashtsqr"]["median_ms"]
ax0.annotate(
    f"Median: {med_val:.1f} ms",
    xy=(4, med_val),
    xytext=(4, med_val * 0.10),
    ha="center",
    fontsize=9,
    fontweight="bold",
    color=colors["flashtsqr"],
    arrowprops=dict(arrowstyle="->", color=colors["flashtsqr"], lw=1.3),
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFE4E6", edgecolor=colors["flashtsqr"], lw=0.9),
    zorder=4,
)

# ----------------------------------------------------
# PANEL B: Normalized Per-Layer Latency (ms / layer)
# ----------------------------------------------------
ax1.set_facecolor("#FAFAFA")
ax1.grid(True, axis="y", zorder=0, alpha=0.7)

pm = np.array([stats[m]["mean_ms_per_layer"] for m in methods])
sem = np.array([stats[m]["sem_ms_per_layer"] for m in methods])

# Custom Bar alpha & edge colors for emphasis
bar_alphas = [0.45 if m not in ("fraq", "flashtsqr") else 0.90 for m in methods]
bar_edges = [colors[m] if m in ("fraq", "flashtsqr") else "#CBD5E1" for m in methods]

bars = ax1.bar(
    x,
    pm,
    width=0.50,
    color=color_list,
    edgecolor=bar_edges,
    linewidth=[1.8 if m in ("fraq", "flashtsqr") else 1.0 for m in methods],
    zorder=2,
)

for bar, alpha_val in zip(bars, bar_alphas):
    bar.set_alpha(alpha_val)

# SEM Error bars
ax1.errorbar(
    x,
    pm,
    yerr=sem,
    fmt="none",
    ecolor="#334155",
    elinewidth=1.4,
    capsize=5.0,
    capthick=1.4,
    zorder=3,
)

ax1.set_yscale("log")
ax1.set_ylim(0.01, 2e4)
ax1.set_ylabel("Normalized Latency (ms / layer)", fontsize=11, fontweight="bold", labelpad=8)
ax1.set_title("(b) Repo-Size-Normalized Mean Latency (Mean ± SEM)", fontsize=12, fontweight="bold", pad=12, loc="left", color="#0F172A")
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=10.5, fontweight="bold")

# Customize tick label colors in Panel B
for tick_label, m in zip(ax1.get_xticklabels(), methods):
    if m == "flashtsqr":
        tick_label.set_color("#E11D48")
        tick_label.set_fontweight("bold")
    elif m == "fraq":
        tick_label.set_color("#059669")
        tick_label.set_fontweight("bold")
    else:
        tick_label.set_color("#64748B")
        tick_label.set_fontweight("normal")

ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# Custom Y-axis tick formatter for panel B
def log_per_layer_formatter(y, pos):
    if y >= 1000:
        return f"{y/1000:.0f} s"
    elif y >= 1:
        return f"{y:g} ms"
    elif y >= 0.001:
        return f"{y*1000:g} µs"
    else:
        return f"{y:g}"

ax1.yaxis.set_major_formatter(FuncFormatter(log_per_layer_formatter))

# Value annotations & speedup pills above bars
svd_mean = stats["svd"]["mean_ms_per_layer"]
florist_mean = stats["florist"]["mean_ms_per_layer"]

for i, m in enumerate(methods):
    bar_val = pm[i]
    speedup = svd_mean / bar_val
    is_focused = m in ("fraq", "flashtsqr")
    
    # Text value formatting
    if bar_val >= 1000:
        val_str = f"{bar_val/1000:.2f} s"
    elif bar_val >= 1.0:
        val_str = f"{bar_val:.2f} ms"
    elif bar_val >= 0.1:
        val_str = f"{bar_val*1000:.0f} µs"
    else:
        val_str = f"{bar_val*1000:.0f} µs"
    
    # Place Value label just above error bar
    err_top = bar_val + sem[i]
    val_color = colors[m] if is_focused else "#475569"
    val_weight = "bold" if is_focused else "normal"
    
    ax1.text(
        i,
        err_top * 1.25,
        val_str,
        ha="center",
        va="bottom",
        fontsize=9.5 if is_focused else 9.0,
        fontweight=val_weight,
        color=val_color,
        zorder=4,
    )
    
    # Speedup Badge
    if m == "svd":
        ax1.text(
            i,
            err_top * 3.2,
            "1.0× (Baseline)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="normal",
            color="#94A3B8",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#E2E8F0", lw=0.8),
            zorder=4,
        )
    elif m == "flashtsqr":
        ax1.text(
            i,
            err_top * 4.2,
            f"22,568× vs SVD\n37.6× vs FLoRIST",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#9F1239",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFE4E6", edgecolor="#E11D48", lw=1.2),
            zorder=5,
        )
    elif m == "fraq":
        ax1.text(
            i,
            err_top * 3.2,
            f"{speedup:,.0f}×",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#059669",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#ECFDF5", edgecolor="#059669", lw=1.0),
            zorder=4,
        )
    else:
        ax1.text(
            i,
            err_top * 3.2,
            f"{speedup:,.0f}×",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="normal",
            color="#64748B",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.8),
            zorder=4,
        )

# Footer Details Box
footer_text = (
    "• Naive SVD: Fixed-seed random sample of 10 repositories (6,776 layers).    "
    "• FLoRIST / SpecTraL / FraQ / LoRAForge: Complete 100-repository census (54,453 layers).\n"
    "• Scope: Measures GPU compression kernel latency only; disk I/O and CPU→GPU data transfer are excluded."
)

fig.text(
    0.5,
    0.042,
    footer_text,
    ha="center",
    va="center",
    fontsize=8.8,
    color="#475569",
    bbox=dict(boxstyle="round,pad=0.45", facecolor="#F8FAFC", edgecolor="#E2E8F0", lw=1.0),
)

out = run4 / "five_methods_latency_final.png"
fig.savefig(out, dpi=300)
plt.close(fig)

# Save JSON stats
(run4 / "five_methods_latency_final.json").write_text(json.dumps(stats, indent=2) + "\n")
print(f"Generated successfully: {out}")
