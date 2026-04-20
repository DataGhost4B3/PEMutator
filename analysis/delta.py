"""
pemutator.analysis.delta
------------------------
Feature-delta utilities: compare original and mutated feature vectors
to understand *which* features actually changed and whether those
changes drove prediction shifts.

Key insight from the experiments
---------------------------------
Large changes in the full feature vector do NOT necessarily translate
to prediction changes.  Only specific features crossing implicit
model thresholds matter.  This module helps identify those features.

Feature group layout (EMBER / gym-malware pefeatures, ~2350 dims)
-----------------------------------------------------------------
Indices are approximate — exact boundaries depend on the pefeatures
version.  The groups below match the try*.py experiments.

  [0  : 100)   header    — PE header metadata (incl. file size at idx 0)
  [100: 500)   section   — Section metadata (names, sizes, characteristics)
  [500:1000)   imports   — Import / export table metadata
  [1000:2000)  histogram — Byte histogram
  [2000:  ∞)   other     — 2D byte-entropy histogram & misc.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Feature group boundaries (as used in try5.py and try6.py)
# ---------------------------------------------------------------------------

FEATURE_GROUPS: dict[str, tuple[int, int]] = {
    "header":    (0,    100),
    "section":   (100,  500),
    "imports":   (500,  1000),
    "histogram": (1000, 2000),
    "other":     (2000, None),   # None → end of vector
}
"""
Coarse groupings of the ~2350-dimensional EMBER feature vector.
Boundaries are inclusive-start, exclusive-end (Python slice convention).
"""

# Specific indices that repeatedly emerged as dominant in the experiments
IMPORTANT_IDX: list[int] = [0, 1, 2, 73, 140, 205, 256, 257]
"""
Feature indices that the try*.py experiments identified as frequently
driving prediction changes.  Index 0 behaves like a file-size metric
(increases linearly with appended bytes).  Index 257 steps discretely
at alignment boundaries.
"""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def feature_delta(
    feat_orig: np.ndarray,
    feat_mut:  np.ndarray,
) -> np.ndarray:
    """
    Compute the element-wise absolute difference between two feature vectors.

    Parameters
    ----------
    feat_orig : np.ndarray, shape (n_features,)
    feat_mut  : np.ndarray, shape (n_features,)

    Returns
    -------
    np.ndarray, shape (n_features,)  — absolute per-feature change.
    """
    return np.abs(feat_orig - feat_mut)


def dominant_features(
    delta: np.ndarray,
    top_k: int = 10,
    indices: list[int] | None = None,
) -> dict:
    """
    Identify the features that changed most.

    Parameters
    ----------
    delta : np.ndarray
        Absolute feature delta (from ``feature_delta``).
    top_k : int
        Return the top-k indices sorted by delta magnitude.
    indices : list[int] or None
        If given, also report delta values for these specific indices
        (e.g. IMPORTANT_IDX from the experiments).

    Returns
    -------
    dict with keys:
      "top_indices"   np.ndarray — indices of largest-delta features (desc.)
      "top_values"    np.ndarray — corresponding delta values
      "named_deltas"  dict[int, float] — delta for each index in ``indices``
      "dominant_idx"  int — single index with the largest delta
                            (among ``indices`` if given, else globally)
    """
    sorted_idx = np.argsort(delta)[::-1]
    top_indices = sorted_idx[:top_k]
    top_values  = delta[top_indices]

    named_deltas: dict[int, float] = {}
    if indices is not None:
        named_deltas = {i: float(delta[i]) for i in indices}
        dominant_idx = int(max(named_deltas, key=named_deltas.get))
    else:
        dominant_idx = int(top_indices[0])

    return {
        "top_indices":   top_indices,
        "top_values":    top_values,
        "named_deltas":  named_deltas,
        "dominant_idx":  dominant_idx,
    }


def group_deltas(
    delta: np.ndarray,
    groups: dict[str, tuple[int, int]] | None = None,
) -> dict[str, float]:
    """
    Aggregate feature deltas by named group (sum of absolute changes).

    Parameters
    ----------
    delta : np.ndarray, shape (n_features,)
        Absolute per-feature change.
    groups : dict or None
        Group name → (start, end) index pair.  Defaults to FEATURE_GROUPS.

    Returns
    -------
    dict[str, float] — total delta summed within each group.
    """
    groups = groups or FEATURE_GROUPS
    result: dict[str, float] = {}
    for name, (start, end) in groups.items():
        slc = delta[start:end]
        result[name] = float(slc.sum())
    return result


def sensitivity_report(
    feat_orig:    np.ndarray,
    feat_mut:     np.ndarray,
    score_before: dict[str, float],
    score_after:  dict[str, float],
    threshold:    float = 1e-4,
) -> dict:
    """
    Full sensitivity report for a single (original, mutated) pair.

    Combines feature-delta analysis with prediction-change detection.

    Parameters
    ----------
    feat_orig, feat_mut : np.ndarray
        Feature vectors before and after mutation.
    score_before, score_after : dict[str, float]
        Malicious-class probability for each model, e.g.
        {"GBDT": 0.82, "RF": 0.60}.
    threshold : float
        Minimum abs change in probability to count as "changed".

    Returns
    -------
    dict with keys:
      "delta"           np.ndarray
      "group_deltas"    dict[str, float]
      "dominant"        dict  (from dominant_features)
      "score_changes"   dict[str, dict] — per-model {before, after, changed}
    """
    delta = feature_delta(feat_orig, feat_mut)

    score_changes: dict[str, dict] = {}
    for name in score_before:
        before = score_before[name]
        after  = score_after.get(name, float("nan"))
        score_changes[name] = {
            "before":  before,
            "after":   after,
            "delta":   abs(after - before),
            "changed": abs(after - before) > threshold,
        }

    return {
        "delta":         delta,
        "group_deltas":  group_deltas(delta),
        "dominant":      dominant_features(delta, indices=IMPORTANT_IDX),
        "score_changes": score_changes,
    }
