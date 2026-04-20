"""
pemutator.viz.evasion_plots
---------------------------
Plotting helpers specific to evasion and fingerprint results.

Functions
---------
plot_evasion_trajectory      — Score over steps for one EvasionResult
plot_campaign_summary        — Bar chart of evasion rates across models
plot_score_waterfall         — Waterfall chart of per-step score drops
plot_fingerprint_bar         — Top-N feature importances (model vs empirical)
plot_attack_surface          — Venn-style overlap scatter
plot_rank_comparison         — Rank scatter plot comparing two models
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from pemutator.analysis.evasion     import EvasionResult
from pemutator.analysis.fingerprint import ModelFingerprint

MODEL_COLORS = {"GBDT": "#2196F3", "RF": "#FF5722"}
STEP_PALETTE = [
    "#E53935", "#8E24AA", "#1E88E5",
    "#00897B", "#F4511E", "#6D4C41",
]


# ---------------------------------------------------------------------------
# 1. Score trajectory for a single evasion run
# ---------------------------------------------------------------------------

def plot_evasion_trajectory(
    result:  EvasionResult,
    figsize: tuple = (10, 5),
) -> tuple:
    """
    Line chart of the malicious-class score over evasion steps.

    Marks the target threshold and annotates each step with the mutation
    name that was applied.

    Parameters
    ----------
    result  : EvasionResult — from GreedyEvasion.run or RandomEvasion.run
    figsize : tuple

    Returns
    -------
    (fig, ax)
    """
    traj = result.trajectory
    xs   = list(range(len(traj)))

    fig, ax = plt.subplots(figsize=figsize)

    color = MODEL_COLORS.get(result.model_name, "#555")
    ax.plot(xs, traj, color=color, linewidth=2.5, marker="o",
            markersize=7, zorder=3, label=f"{result.model_name} score")

    ax.axhline(result.target, color="red", linestyle="--", linewidth=1.2,
               label=f"target = {result.target}")
    ax.axhline(result.initial_score, color=color, linestyle=":",
               linewidth=1, alpha=0.5, label="baseline")

    # Annotate mutation names at each committed step
    for i, step in enumerate(result.steps):
        step_x = i + 1          # trajectory[0] = baseline, steps shift by 1
        if step_x < len(traj):
            ax.annotate(
                step.mutation,
                xy=(step_x, traj[step_x]),
                xytext=(0, 12), textcoords="offset points",
                ha="center", fontsize=8,
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8),
            )

    status = "EVADED ✓" if result.evaded else "not evaded"
    ax.set_title(
        f"{result.strategy.title()} Evasion — {result.model_name} — {status}\n"
        f"{result.path}",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlabel("Evasion step", fontsize=11)
    ax.set_ylabel("P(malicious)", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(xs)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 2. Campaign summary bar chart
# ---------------------------------------------------------------------------

def plot_campaign_summary(
    summary: dict,
    figsize: tuple = (9, 5),
) -> tuple:
    """
    Grouped bar chart: evasion rate and average score drop per model.

    Parameters
    ----------
    summary : dict — from evasion_summary()
    figsize : tuple

    Returns
    -------
    (fig, axes) — two side-by-side axes
    """
    model_names = list(summary.keys())
    rates  = [summary[n]["evasion_rate"]  for n in model_names]
    drops  = [summary[n]["avg_score_drop"] for n in model_names]
    colors = [MODEL_COLORS.get(n, "gray") for n in model_names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle("Evasion Campaign Summary", fontsize=13, fontweight="bold")

    # evasion rate
    bars1 = ax1.bar(model_names, rates, color=colors, edgecolor="k",
                    linewidth=0.7, alpha=0.85)
    for bar, r in zip(bars1, rates):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{r:.1%}", ha="center", va="bottom", fontsize=11)
    ax1.set_ylim(0, 1.2)
    ax1.set_ylabel("Evasion rate", fontsize=11)
    ax1.set_title("Evasion rate", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")

    # avg score drop
    bars2 = ax2.bar(model_names, drops, color=colors, edgecolor="k",
                    linewidth=0.7, alpha=0.85)
    for bar, d in zip(bars2, drops):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{d:.3f}", ha="center", va="bottom", fontsize=11)
    ax2.set_ylim(0, max(drops) * 1.3 if drops else 1)
    ax2.set_ylabel("Avg score drop", fontsize=11)
    ax2.set_title("Avg score drop", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------------
# 3. Score waterfall (per-step contributions)
# ---------------------------------------------------------------------------

def plot_score_waterfall(
    result:  EvasionResult,
    figsize: tuple = (10, 5),
) -> tuple:
    """
    Waterfall chart showing the cumulative score change per mutation step.

    Each bar shows the delta for that step; positive (red) = score went up,
    negative (green) = score went down (good for evasion).

    Parameters
    ----------
    result  : EvasionResult
    figsize : tuple

    Returns
    -------
    (fig, ax)
    """
    steps    = result.steps
    if not steps:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No steps recorded", ha="center", transform=ax.transAxes)
        return fig, ax

    labels   = [f"Step {s.step}\n{s.mutation}" for s in steps]
    deltas   = [s.score_delta for s in steps]
    running  = result.initial_score

    fig, ax = plt.subplots(figsize=figsize)

    bottoms = []
    for d in deltas:
        bottoms.append(running)
        running += d

    for i, (label, d, bottom) in enumerate(zip(labels, deltas, bottoms)):
        color = "#4CAF50" if d <= 0 else "#F44336"
        ax.bar(i, abs(d), bottom=min(bottom, bottom + d),
               color=color, edgecolor="k", linewidth=0.7, alpha=0.85)
        ax.text(i, bottom + d + (0.01 if d >= 0 else -0.03),
                f"{d:+.3f}", ha="center", va="bottom" if d >= 0 else "top",
                fontsize=9)

    ax.axhline(result.initial_score, color="black", linestyle="--",
               linewidth=1, label=f"baseline = {result.initial_score:.3f}")
    ax.axhline(result.target, color="red", linestyle=":",
               linewidth=1.2, label=f"target = {result.target}")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("P(malicious)", fontsize=11)
    ax.set_title(
        f"Score waterfall — {result.model_name} — {result.strategy}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 4. Feature importance bar chart
# ---------------------------------------------------------------------------

def plot_fingerprint_bar(
    fp:             ModelFingerprint,
    empirical_imp:  np.ndarray | None = None,
    top_k:          int   = 30,
    figsize:        tuple = (13, 5),
) -> tuple:
    """
    Bar chart of top-k feature importances for a model.
    Optionally overlays empirical importance as a step line.

    Parameters
    ----------
    fp             : ModelFingerprint
    empirical_imp  : np.ndarray or None — from empirical_importance()
    top_k          : int
    figsize        : tuple

    Returns
    -------
    (fig, ax)
    """
    top_idx  = fp.top_indices(top_k)
    top_vals = fp.importances[top_idx]

    fig, ax = plt.subplots(figsize=figsize)
    color = MODEL_COLORS.get(fp.model_name, "#2196F3")
    xs = np.arange(top_k)

    ax.bar(xs, top_vals, color=color, alpha=0.75, edgecolor="k",
           linewidth=0.5, label=f"{fp.model_name} importance")

    # Highlight IMPORTANT_IDX from experiments
    from pemutator.analysis.delta import IMPORTANT_IDX
    for i, idx in enumerate(top_idx):
        if idx in IMPORTANT_IDX:
            ax.bar(i, top_vals[i], color="gold", alpha=0.9,
                   edgecolor="k", linewidth=0.7)

    if empirical_imp is not None:
        emp_vals_norm = empirical_imp[top_idx]
        if emp_vals_norm.max() > 0:
            emp_vals_norm = emp_vals_norm / emp_vals_norm.max() * top_vals.max()
        ax2 = ax.twinx()
        ax2.step(xs, emp_vals_norm, where="mid", color="#FF5722",
                 linewidth=1.8, label="empirical (normalised)")
        ax2.set_ylabel("Empirical importance (normalised)", fontsize=10)
        ax2.legend(loc="upper right", fontsize=9)
        ax2.set_ylim(0)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in top_idx], rotation=90, fontsize=7)
    ax.set_xlabel("Feature index", fontsize=11)
    ax.set_ylabel("Feature importance", fontsize=11)
    ax.set_title(
        f"Top-{top_k} features — {fp.model_name}  "
        f"(gold = experiment-identified indices)",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 5. Attack surface scatter
# ---------------------------------------------------------------------------

def plot_attack_surface(
    surface: dict,
    fp:      ModelFingerprint,
    emp_imp: np.ndarray,
    figsize: tuple = (8, 6),
) -> tuple:
    """
    Scatter plot of model importance vs empirical importance for all features,
    highlighting the attack-surface overlap.

    Parameters
    ----------
    surface : dict — from attack_surface()
    fp      : ModelFingerprint
    emp_imp : np.ndarray
    figsize : tuple

    Returns
    -------
    (fig, ax)
    """
    n   = len(fp.importances)
    xi  = fp.importances
    yi  = emp_imp[:n] if len(emp_imp) >= n else np.pad(emp_imp, (0, n - len(emp_imp)))

    overlap_set = set(surface["overlap"].tolist())
    reachable   = set(surface["reachable"].tolist())
    model_top   = set(surface["model_top"].tolist())

    colors = []
    for idx in range(n):
        if idx in overlap_set:
            colors.append("#E91E63")     # overlap = pink
        elif idx in reachable:
            colors.append("#FF9800")     # reachable only = orange
        elif idx in model_top:
            colors.append("#2196F3")     # model-important only = blue
        else:
            colors.append("#BDBDBD")     # neither = grey

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(xi, yi, c=colors, alpha=0.6, s=18, linewidths=0)

    # Legend patches
    patches = [
        mpatches.Patch(color="#E91E63", label=f"Attack surface (n={len(overlap_set)})"),
        mpatches.Patch(color="#FF9800", label="Reachable only"),
        mpatches.Patch(color="#2196F3", label="Model-important only"),
        mpatches.Patch(color="#BDBDBD", label="Neither"),
    ]
    ax.legend(handles=patches, fontsize=9)
    ax.set_xlabel(f"{fp.model_name} feature importance", fontsize=11)
    ax.set_ylabel("Empirical importance", fontsize=11)
    ax.set_title(
        f"Attack surface — {fp.model_name}\n"
        f"Overlap: {len(overlap_set)} of {len(model_top)} model-top features reachable by mutations",
        fontsize=11,
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 6. Rank comparison scatter (GBDT vs RF)
# ---------------------------------------------------------------------------

def plot_rank_comparison(
    comparison: dict,
    fp_a:       ModelFingerprint,
    fp_b:       ModelFingerprint,
    top_k:      int   = 50,
    figsize:    tuple = (7, 7),
) -> tuple:
    """
    Scatter plot of feature rank in model A vs model B.

    Points near the diagonal → both models agree this feature matters.
    Points far from the diagonal → architecture-specific sensitivity.

    Parameters
    ----------
    comparison : dict — from compare_fingerprints()
    fp_a, fp_b : ModelFingerprint
    top_k      : int — only plot features in the union top-k
    figsize    : tuple

    Returns
    -------
    (fig, ax)
    """
    union   = comparison["union"]
    name_a  = fp_a.model_name
    name_b  = fp_b.model_name
    rm      = comparison["rank_matrix"]

    ranks_a = [rm[int(i)][name_a] for i in union]
    ranks_b = [rm[int(i)][name_b] for i in union]
    inter   = set(comparison["intersection"].tolist())
    is_both = [int(i) in inter for i in union]

    colors = ["#E91E63" if b else "#90CAF9" for b in is_both]

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(ranks_a, ranks_b, c=colors, alpha=0.7, s=30, linewidths=0.3,
               edgecolors="k")
    lim = max(max(ranks_a), max(ranks_b)) + 5
    ax.plot([1, lim], [1, lim], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel(f"Rank in {name_a}", fontsize=11)
    ax.set_ylabel(f"Rank in {name_b}", fontsize=11)
    ax.set_title(
        f"Feature rank agreement — {name_a} vs {name_b}\n"
        f"(pink = in both top-{top_k}; blue = union only)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig, ax
