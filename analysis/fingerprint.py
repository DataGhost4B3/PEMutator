"""
pemutator.analysis.fingerprint
-------------------------------
Feature importance attribution and "effective feature set" analysis.

The sweep and delta experiments revealed that classifiers rely on a
**sparse subset** of the ~2350-dimensional feature vector.  This module
makes that sparsity explicit by:

1. Extracting built-in feature importances from tree ensembles.
2. Computing empirical importances from the mutation experiments
   (which features actually moved when predictions changed?).
3. Cross-referencing to find features that are both model-important
   AND mutation-reachable (the "attack surface").

Classes / Functions
-------------------
ModelFingerprint
    Wraps a fitted sklearn tree ensemble and exposes feature importance
    rankings, group-level importance summaries, and overlap analysis.

empirical_importance(sweep_records, ...)
    Derive importance scores from sweep data: features that changed at
    the moment of a prediction shift get weighted credit.

attack_surface(model_fp, empirical_imp, top_k)
    Intersect model importance and empirical reachability to identify
    the features that are both important to the model and reachable
    by the mutations we have implemented.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from pemutator.analysis.delta import FEATURE_GROUPS, IMPORTANT_IDX


# ---------------------------------------------------------------------------
# ModelFingerprint
# ---------------------------------------------------------------------------

class ModelFingerprint:
    """
    Feature importance analysis for a fitted sklearn tree ensemble.

    Parameters
    ----------
    model : BaseEstimator
        A fitted GradientBoostingClassifier or RandomForestClassifier
        (anything with a ``feature_importances_`` attribute).
    model_name : str
        Label used in display/reports.
    n_features : int, optional
        Total feature vector length.  Inferred from the model if not given.
    """

    def __init__(
        self,
        model:      BaseEstimator,
        model_name: str = "model",
        n_features: int | None = None,
    ):
        if not hasattr(model, "feature_importances_"):
            raise ValueError(
                f"{type(model).__name__} does not expose feature_importances_. "
                "Only fitted tree ensembles are supported."
            )
        self.model      = model
        self.model_name = model_name
        self.importances: np.ndarray = model.feature_importances_
        self.n_features  = n_features or len(self.importances)

    # ------------------------------------------------------------------
    # Rankings
    # ------------------------------------------------------------------

    def top_indices(self, k: int = 20) -> np.ndarray:
        """Return indices of the top-k most important features (descending)."""
        return np.argsort(self.importances)[::-1][:k]

    def top_importances(self, k: int = 20) -> np.ndarray:
        """Return the importance values for the top-k features."""
        return self.importances[self.top_indices(k)]

    def rank_of(self, idx: int) -> int:
        """Return the rank (1-based) of a specific feature index."""
        sorted_idx = np.argsort(self.importances)[::-1]
        rank = int(np.where(sorted_idx == idx)[0][0]) + 1
        return rank

    # ------------------------------------------------------------------
    # Group-level summaries
    # ------------------------------------------------------------------

    def group_importance(
        self,
        groups: dict[str, tuple[int, int]] | None = None,
    ) -> dict[str, float]:
        """
        Sum of feature importances within each named group.

        Returns
        -------
        dict[str, float] — normalised to sum to 1 across groups.
        """
        groups = groups or FEATURE_GROUPS
        raw: dict[str, float] = {}
        for name, (start, end) in groups.items():
            raw[name] = float(self.importances[start:end].sum())
        total = sum(raw.values()) or 1.0
        return {k: v / total for k, v in raw.items()}

    # ------------------------------------------------------------------
    # Specific-index reporting
    # ------------------------------------------------------------------

    def report_indices(self, indices: list[int] | None = None) -> dict[int, dict]:
        """
        For each index in ``indices``, report its importance and rank.

        Parameters
        ----------
        indices : list[int] or None
            Defaults to IMPORTANT_IDX from the experiments.

        Returns
        -------
        dict[int, {"importance": float, "rank": int, "percentile": float}]
        """
        indices = indices or IMPORTANT_IDX
        out: dict[int, dict] = {}
        sorted_desc = np.argsort(self.importances)[::-1]
        n = len(self.importances)
        for idx in indices:
            rank = int(np.where(sorted_desc == idx)[0][0]) + 1
            out[idx] = {
                "importance": float(self.importances[idx]),
                "rank":       rank,
                "percentile": round(100.0 * (1 - rank / n), 2),
            }
        return out

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        top3 = self.top_indices(3)
        return (
            f"ModelFingerprint({self.model_name}, "
            f"n_features={self.n_features}, "
            f"top3={list(top3)})"
        )


# ---------------------------------------------------------------------------
# Empirical importance from sweep data
# ---------------------------------------------------------------------------

def empirical_importance(
    sweep_records: list[dict],
    scores_orig:   dict[str, float],
    model_name:    str,
    threshold:     float = 0.02,
    n_features:    int   = 2350,
) -> np.ndarray:
    """
    Derive a feature importance vector from size-sweep records.

    At each step where the prediction changed significantly, the features
    that changed most are credited proportionally to the size of their delta.

    This produces an *empirical* importance — "which features moved when
    the model changed its mind?" — as opposed to the model's built-in
    feature_importances_ (which reflect training-time splits).

    Parameters
    ----------
    sweep_records : list[dict]
        The ``records`` list from size_sweep output.
    scores_orig : dict[str, float]
        Baseline scores, e.g. sweep_result["scores_orig"].
    model_name : str
        Which model's score to track.
    threshold : float
        Minimum absolute score change to count as a "change event".
    n_features : int
        Length of the feature vector.

    Returns
    -------
    np.ndarray, shape (n_features,)
        Unnormalised empirical importance scores.
    """
    importance = np.zeros(n_features)
    prev_score = scores_orig[model_name]
    prev_feat  = None

    for rec in sweep_records:
        curr_score = rec["scores"][model_name]
        curr_feat  = rec["feat_mut"]

        if prev_feat is not None and abs(curr_score - prev_score) > threshold:
            delta = np.abs(curr_feat - prev_feat)
            # Credit features proportionally to their delta at the change event
            importance += delta / (delta.sum() + 1e-12)

        prev_score = curr_score
        prev_feat  = curr_feat

    return importance


# ---------------------------------------------------------------------------
# Attack surface intersection
# ---------------------------------------------------------------------------

def attack_surface(
    model_fp:       ModelFingerprint,
    empirical_imp:  np.ndarray,
    top_k:          int = 30,
) -> dict:
    """
    Find the intersection of model-important features and
    mutation-reachable features.

    A feature is "model-important" if it is in the model's top-k.
    A feature is "mutation-reachable" if it has non-zero empirical importance
    (i.e. it changed during at least one prediction-flip event).

    Parameters
    ----------
    model_fp      : ModelFingerprint
    empirical_imp : np.ndarray — from empirical_importance()
    top_k         : int

    Returns
    -------
    dict with keys:
      "model_top"       np.ndarray — top-k model importance indices
      "reachable"       np.ndarray — indices with empirical_imp > 0
      "overlap"         np.ndarray — intersection
      "overlap_size"    int
      "overlap_scores"  dict[int, {"model_importance", "empirical_importance"}]
    """
    model_top  = model_fp.top_indices(top_k)
    reachable  = np.where(empirical_imp > 0)[0]
    overlap    = np.intersect1d(model_top, reachable)

    overlap_scores: dict[int, dict] = {}
    for idx in overlap:
        overlap_scores[int(idx)] = {
            "model_importance":    float(model_fp.importances[idx]),
            "empirical_importance": float(empirical_imp[idx]),
        }

    return {
        "model_top":      model_top,
        "reachable":      reachable,
        "overlap":        overlap,
        "overlap_size":   len(overlap),
        "overlap_scores": overlap_scores,
    }


# ---------------------------------------------------------------------------
# Cross-model comparison
# ---------------------------------------------------------------------------

def compare_fingerprints(
    fps: dict[str, ModelFingerprint],
    top_k: int = 50,
) -> dict:
    """
    Compare feature importance rankings across multiple models.

    Parameters
    ----------
    fps : dict[str, ModelFingerprint]
    top_k : int — how many top features to compare

    Returns
    -------
    dict with keys:
      "union"       np.ndarray — union of all models' top-k sets
      "intersection" np.ndarray — features in ALL models' top-k
      "per_model"   dict[str, np.ndarray] — each model's top-k
      "rank_matrix" dict[int, dict[str, int]] — rank of each union feature per model
    """
    per_model  = {name: fp.top_indices(top_k) for name, fp in fps.items()}
    union_set  = set()
    inter_set  = None

    for indices in per_model.values():
        s = set(indices.tolist())
        union_set |= s
        inter_set = s if inter_set is None else (inter_set & s)

    inter_set = inter_set or set()

    rank_matrix: dict[int, dict[str, int]] = {}
    for feat_idx in union_set:
        rank_matrix[feat_idx] = {}
        for name, fp in fps.items():
            rank_matrix[feat_idx][name] = fp.rank_of(feat_idx)

    return {
        "union":        np.array(sorted(union_set)),
        "intersection": np.array(sorted(inter_set)),
        "per_model":    per_model,
        "rank_matrix":  rank_matrix,
    }
