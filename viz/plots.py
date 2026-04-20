"""
pemutator.viz.plots
-------------------
Matplotlib-based plotting helpers for visualising sweep and
sensitivity results in a Jupyter notebook.

All functions return (fig, ax) tuples so the caller can further
customise the plot.  They do NOT call plt.show() — the notebook
will display figures automatically.

Functions
---------
plot_size_sweep          — Feature values + prediction score vs appended bytes
plot_feature_trajectory  — Multi-index feature values across the sweep
plot_mutation_sensitivity — Bar chart of sensitivity rates per mutation/model
plot_group_deltas        — Stacked bar of group-level feature changes
plot_score_heatmap       — Heatmap of scores over (file, mutation) grid
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Colour palette (consistent across all plots)
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "GBDT": "#2196F3",   # blue
    "RF":   "#FF5722",   # deep orange
}

MUTATION_COLORS = {
    "append":  "#4CAF50",
    "import":  "#9C27B0",
    "header":  "#FF9800",
    "section": "#F44336",
}

GROUP_COLORS = {
    "header":    "#3F51B5",
    "section":   "#009688",
    "imports":   "#FF5722",
    "histogram": "#FFC107",
    "other":     "#9E9E9E",
}


# ---------------------------------------------------------------------------
# 1. Size sweep overview
# ---------------------------------------------------------------------------

def plot_size_sweep(
    sweep_result: dict,
    model_name:   str = "RF",
    track_idx:    list[int] | None = None,
    figsize:      tuple = (14, 8),
) -> tuple:
    """
    Two-panel plot for a size_sweep result:
      Top    — tracked feature values vs bytes appended
      Bottom — model score vs bytes appended, with first-change marked

    Parameters
    ----------
    sweep_result : dict
        Output of pemutator.analysis.sweep.size_sweep.
    model_name : str
        Which model to show in the score panel.
    track_idx : list[int] or None
        Feature indices to plot in the top panel.  Defaults to [0, 257].
    figsize : tuple

    Returns
    -------
    (fig, axes) — matplotlib figure and array of axes.
    """
    track_idx = track_idx or [0, 257]
    records   = sweep_result["records"]
    first_ch  = sweep_result["first_change"].get(model_name)

    ns     = [r["n"]               for r in records]
    scores = [r["scores"][model_name] for r in records]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(
        f"Size Sweep — {sweep_result['path']}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # ---- top: feature trajectories ----
    ax0 = axes[0]
    colors_cycle = plt.cm.tab10(np.linspace(0, 0.9, len(track_idx)))
    for idx, col in zip(track_idx, colors_cycle):
        vals = [r["tracked"].get(idx, np.nan) for r in records]
        ax0.plot(ns, vals, label=f"feat[{idx}]", color=col, linewidth=1.8)

    ax0.set_ylabel("Feature value", fontsize=11)
    ax0.legend(fontsize=10)
    ax0.grid(True, alpha=0.3)
    ax0.set_title("Tracked feature values", fontsize=11)

    # ---- bottom: prediction score ----
    ax1 = axes[1]
    color = MODEL_COLORS.get(model_name, "#333")
    ax1.plot(ns, scores, color=color, linewidth=2, label=f"{model_name} score")
    ax1.axhline(
        sweep_result["scores_orig"][model_name],
        color=color, linestyle="--", alpha=0.5, linewidth=1,
        label="baseline",
    )

    if first_ch is not None:
        ax1.axvline(
            first_ch["size"], color="red", linestyle=":", linewidth=1.5,
            label=f"1st change @ {first_ch['size']}B\n"
                  f"(dominant feat[{first_ch['dominant_idx']}])",
        )

    ax1.set_xlabel("Bytes appended", fontsize=11)
    ax1.set_ylabel("P(malicious)", fontsize=11)
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f"{model_name} prediction", fontsize=11)

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 2. Multi-index feature trajectory
# ---------------------------------------------------------------------------

def plot_feature_trajectory(
    sweep_result: dict,
    indices:      list[int],
    figsize:      tuple = (13, 5),
) -> tuple:
    """
    Plot the value of each feature index across the size sweep.

    Useful for visually confirming that some features (e.g. index 0) change
    linearly while others (e.g. index 257) step discretely.

    Parameters
    ----------
    sweep_result : dict   — from size_sweep
    indices      : list[int]
    figsize      : tuple

    Returns
    -------
    (fig, ax)
    """
    records = sweep_result["records"]
    ns = [r["n"] for r in records]

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab20(np.linspace(0, 1, len(indices)))

    for idx, col in zip(indices, colors):
        vals = [float(r["feat_mut"][idx]) for r in records]
        ax.plot(ns, vals, label=f"feat[{idx}]", color=col, linewidth=1.6)

    ax.set_xlabel("Bytes appended", fontsize=11)
    ax.set_ylabel("Feature value", fontsize=11)
    ax.set_title("Feature trajectories across size sweep", fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 3. Mutation sensitivity bar chart
# ---------------------------------------------------------------------------

def plot_mutation_sensitivity(
    sens_result: dict,
    figsize:     tuple = (10, 5),
) -> tuple:
    """
    Grouped bar chart: sensitivity rate per mutation type, split by model.

    Parameters
    ----------
    sens_result : dict — from mutation_sensitivity
    figsize : tuple

    Returns
    -------
    (fig, ax)
    """
    totals    = sens_result["totals"]
    mut_names = list(totals.keys())
    mod_names = list(next(iter(totals.values())).keys())
    n_muts    = len(mut_names)
    n_mods    = len(mod_names)

    x    = np.arange(n_muts)
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    for i, mod in enumerate(mod_names):
        rates  = [totals[m][mod] for m in mut_names]
        offset = (i - (n_mods - 1) / 2) * width
        color  = MODEL_COLORS.get(mod, f"C{i}")
        bars   = ax.bar(x + offset, rates, width, label=mod, color=color, alpha=0.85)
        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{rate:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(mut_names, fontsize=11)
    ax.set_ylabel("Sensitivity rate (fraction of files changed)", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Mutation Sensitivity by Model", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Group-level feature delta
# ---------------------------------------------------------------------------

def plot_group_deltas(
    sweep_result: dict,
    figsize:      tuple = (13, 5),
) -> tuple:
    """
    Stacked area chart of group-level feature deltas across the size sweep.

    Shows which feature group absorbs the most change as bytes are appended.

    Parameters
    ----------
    sweep_result : dict — from size_sweep
    figsize : tuple

    Returns
    -------
    (fig, ax)
    """
    records    = sweep_result["records"]
    ns         = [r["n"] for r in records]
    group_keys = list(records[0]["group_deltas"].keys())

    data = {g: [r["group_deltas"][g] for r in records] for g in group_keys}

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(ns))

    for g in group_keys:
        vals   = np.array(data[g])
        color  = GROUP_COLORS.get(g, "gray")
        ax.fill_between(ns, bottom, bottom + vals, alpha=0.7,
                        label=g, color=color)
        bottom += vals

    ax.set_xlabel("Bytes appended", fontsize=11)
    ax.set_ylabel("Cumulative |Δfeature|", fontsize=11)
    ax.set_title("Feature group delta by append size", fontsize=12)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 5. Score heatmap (mutation × file)
# ---------------------------------------------------------------------------

def plot_score_heatmap(
    sens_result: dict,
    model_name:  str   = "RF",
    figsize:     tuple = (12, 6),
) -> tuple:
    """
    Heatmap of prediction score (post-mutation) for each (file, mutation) pair.

    Parameters
    ----------
    sens_result : dict — from mutation_sensitivity
    model_name  : str
    figsize     : tuple

    Returns
    -------
    (fig, ax)
    """
    records   = sens_result["records"]
    mut_names = list({r["mutation"] for r in records})
    file_names = list({r["path"] for r in records})

    # Build score matrix [n_files × n_mutations]
    score_matrix = np.full((len(file_names), len(mut_names)), np.nan)
    idx_f = {f: i for i, f in enumerate(file_names)}
    idx_m = {m: j for j, m in enumerate(mut_names)}

    for r in records:
        i = idx_f[r["path"]]
        j = idx_m[r["mutation"]]
        score_matrix[i, j] = r["scores_mut"][model_name]

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(score_matrix, aspect="auto", cmap="RdYlGn_r",
                   vmin=0, vmax=1)

    ax.set_xticks(range(len(mut_names)))
    ax.set_xticklabels(mut_names, fontsize=10)
    ax.set_yticks(range(len(file_names)))
    ax.set_yticklabels(
        [p.split("/")[-1][:20] for p in file_names],
        fontsize=8,
    )
    ax.set_title(f"{model_name} P(malicious) after mutation", fontsize=12)
    fig.colorbar(im, ax=ax, label="P(malicious)")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 6. Model comparison scatter
# ---------------------------------------------------------------------------

def plot_model_comparison(
    sens_result: dict,
    mut_name:    str   = "append",
    figsize:     tuple = (6, 6),
) -> tuple:
    """
    Scatter plot comparing GBDT vs RF scores (post-mutation) across all files
    for a given mutation type.  Points above the diagonal mean RF predicts
    higher maliciousness than GBDT.

    Parameters
    ----------
    sens_result : dict — from mutation_sensitivity
    mut_name : str     — which mutation to show
    figsize : tuple

    Returns
    -------
    (fig, ax)
    """
    records = [r for r in sens_result["records"] if r["mutation"] == mut_name]
    if not records:
        raise ValueError(f"No records found for mutation '{mut_name}'.")

    mod_names = list(records[0]["scores_mut"].keys())
    if len(mod_names) < 2:
        raise ValueError("Need at least two models for comparison plot.")

    name_a, name_b = mod_names[0], mod_names[1]
    a_vals = [r["scores_mut"][name_a] for r in records]
    b_vals = [r["scores_mut"][name_b] for r in records]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(a_vals, b_vals, alpha=0.7,
               color=MODEL_COLORS.get(name_b, "C1"), edgecolors="k", linewidth=0.5)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="y = x")
    ax.set_xlabel(f"{name_a} P(malicious)", fontsize=11)
    ax.set_ylabel(f"{name_b} P(malicious)", fontsize=11)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Model comparison — mutation: {mut_name}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax
