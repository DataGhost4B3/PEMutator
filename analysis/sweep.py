"""
pemutator.analysis.sweep
------------------------
Sweep experiments: vary perturbation magnitude or type across a sample
and collect structured results.

Two main entry points
---------------------
size_sweep(path, extractor, models, ...)
    Replicate the try7.py / try8.py experiment: incrementally grow the
    number of appended bytes, record feature[0], feature[257] and all
    model scores, detect the first significant prediction change and
    identify the dominant feature at that point.

mutation_sensitivity(paths, extractor, models, ...)
    Replicate try4.py / try5.py: apply all four mutation types to each
    file in a list and summarise how often each mutation triggers a
    prediction change for each model.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from pemutator.core.extractor import FeatureExtractor
from pemutator.core.mutator   import append_bytes, MUTATIONS
from pemutator.analysis.delta  import (
    feature_delta,
    dominant_features,
    group_deltas,
    IMPORTANT_IDX,
)


# ---------------------------------------------------------------------------
# size_sweep
# ---------------------------------------------------------------------------

def size_sweep(
    path:         str,
    extractor:    FeatureExtractor,
    models:       dict[str, BaseEstimator],
    sizes:        list[int] | None = None,
    track_idx:    list[int] | None = None,
    threshold:    float = 0.02,
) -> dict:
    """
    Incrementally append bytes to a PE file and record how features and
    classifier scores evolve.

    This directly reproduces the core experiment of try7.py / try8.py.
    The key findings were:

    * Predictions behave in a piecewise-constant fashion — they stay
      flat for long stretches, then jump abruptly.
    * feature[0] (size-like) increases linearly; feature[257] (alignment)
      steps discretely at ~2048-byte boundaries.
    * The dominant feature at the first prediction change is usually [257].

    Parameters
    ----------
    path : str
        PE file to experiment on.
    extractor : FeatureExtractor
    models : dict[str, BaseEstimator]
    sizes : list[int] or None
        Byte counts to sweep.  Defaults to range(0, 20000, 200).
    track_idx : list[int] or None
        Feature indices to track per step.  Defaults to [0, 257].
    threshold : float
        Minimum absolute prediction change to count as a "significant change".
        Matches the THRESH = 0.02 used in try7.py.

    Returns
    -------
    dict with keys:
      "path"          str
      "feat_orig"     np.ndarray — baseline feature vector
      "scores_orig"   dict[str, float]
      "records"       list[dict] — one entry per size step
      "first_change"  dict[str, dict] — per model: {size, dominant_idx, record}
                      or None if no change was detected
    """
    sizes      = sizes      or list(range(0, 20_000, 200))
    track_idx  = track_idx  or [0, 257]

    # --- baseline ---
    with open(path, "rb") as fh:
        orig_bytes = fh.read()
    feat_orig = extractor.extract(orig_bytes)
    X_orig    = feat_orig.reshape(1, -1)

    scores_orig: dict[str, float] = {
        name: float(clf.predict_proba(X_orig)[0][1])
        for name, clf in models.items()
    }

    records: list[dict] = []
    first_change: dict[str, dict | None] = {name: None for name in models}

    for n in sizes:
        mut_bytes = append_bytes(path, n)
        feat_mut  = extractor.extract(mut_bytes)
        X_mut     = feat_mut.reshape(1, -1)

        delta = feature_delta(feat_orig, feat_mut)
        dom   = dominant_features(delta, indices=IMPORTANT_IDX)

        scores_mut: dict[str, float] = {
            name: float(clf.predict_proba(X_mut)[0][1])
            for name, clf in models.items()
        }

        tracked = {i: float(feat_mut[i]) for i in track_idx}
        grp_d   = group_deltas(delta)

        rec = {
            "n":            n,
            "feat_mut":     feat_mut,
            "scores":       scores_mut,
            "tracked":      tracked,
            "group_deltas": grp_d,
            "dominant_idx": dom["dominant_idx"],
        }
        records.append(rec)

        for name, baseline in scores_orig.items():
            if first_change[name] is None:
                if abs(scores_mut[name] - baseline) > threshold:
                    first_change[name] = {
                        "size":         n,
                        "dominant_idx": dom["dominant_idx"],
                        "record":       rec,
                    }

    return {
        "path":         path,
        "feat_orig":    feat_orig,
        "scores_orig":  scores_orig,
        "records":      records,
        "first_change": first_change,
    }


# ---------------------------------------------------------------------------
# mutation_sensitivity
# ---------------------------------------------------------------------------

def mutation_sensitivity(
    paths:      list[str],
    extractor:  FeatureExtractor,
    models:     dict[str, BaseEstimator],
    mutations:  dict[str, callable] | None = None,
    threshold:  float = 1e-4,
    verbose:    bool  = False,
) -> dict:
    """
    Apply each mutation in ``mutations`` to each file in ``paths`` and
    count how often each mutation triggers a prediction change.

    Reproduces the aggregate experiment of try4.py / try5.py.

    Parameters
    ----------
    paths : list[str]
        PE files to evaluate.
    extractor : FeatureExtractor
    models : dict[str, BaseEstimator]
    mutations : dict[str, callable] or None
        Mutation name → fn(path) → bytes.  Defaults to MUTATIONS.
    threshold : float
        Minimum abs prediction change to count as sensitive.
    verbose : bool
        Print per-file, per-mutation results.

    Returns
    -------
    dict with keys:
      "counts"  dict[mutation_name, dict[model_name, int]]
                — number of files that triggered a change
      "totals"  dict[mutation_name, dict[model_name, float]]
                — sensitivity rate (0–1) = counts / n_files
      "records" list[dict]
                — raw per-file, per-mutation probe records
    """
    mutations = mutations or MUTATIONS
    n_files   = len(paths)

    counts: dict[str, dict[str, int]] = {
        m: {name: 0 for name in models} for m in mutations
    }
    records: list[dict] = []

    for path in paths:
        try:
            with open(path, "rb") as fh:
                orig_bytes = fh.read()
            feat_orig = extractor.extract(orig_bytes)
            X_orig    = feat_orig.reshape(1, -1)

            scores_orig: dict[str, float] = {
                name: float(clf.predict_proba(X_orig)[0][1])
                for name, clf in models.items()
            }

        except Exception as exc:
            if verbose:
                print(f"[SKIP] {path}: {exc}")
            continue

        for mut_name, mut_fn in mutations.items():
            try:
                mut_bytes = mut_fn(path)
                feat_mut  = extractor.extract(mut_bytes)
                X_mut     = feat_mut.reshape(1, -1)

                scores_mut: dict[str, float] = {
                    name: float(clf.predict_proba(X_mut)[0][1])
                    for name, clf in models.items()
                }

                delta   = feature_delta(feat_orig, feat_mut)
                dom     = dominant_features(delta, indices=IMPORTANT_IDX)
                grp_d   = group_deltas(delta)

                changed: dict[str, bool] = {}
                for name in models:
                    diff = abs(scores_mut[name] - scores_orig[name])
                    changed[name] = diff > threshold
                    if changed[name]:
                        counts[mut_name][name] += 1

                rec = {
                    "path":         path,
                    "mutation":     mut_name,
                    "scores_orig":  scores_orig,
                    "scores_mut":   scores_mut,
                    "changed":      changed,
                    "group_deltas": grp_d,
                    "dominant_idx": dom["dominant_idx"],
                }
                records.append(rec)

                if verbose:
                    print(
                        f"{path} | {mut_name:8s} | "
                        + "  ".join(
                            f"{n}: {scores_orig[n]:.3f}→{scores_mut[n]:.3f}"
                            f"({'✓' if changed[n] else '–'})"
                            for n in models
                        )
                    )

            except Exception as exc:
                if verbose:
                    print(f"[ERR] {path} | {mut_name}: {exc}")

    totals: dict[str, dict[str, float]] = {
        m: {name: counts[m][name] / max(n_files, 1) for name in models}
        for m in mutations
    }

    return {
        "counts":  counts,
        "totals":  totals,
        "records": records,
    }
