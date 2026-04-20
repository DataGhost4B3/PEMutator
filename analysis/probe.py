"""
pemutator.analysis.probe
------------------------
Single-sample probing: apply one mutation to one file and return a
structured result.

This is the conceptual unit of the earliest experiments (try.py,
try2.py, try3.py) — apply a mutation, extract features, compare scores.

Example
-------
    from pemutator import FeatureExtractor, build_models, train_models
    from pemutator import probe_sample, append_bytes
    from pemutator.core.models import make_balanced_labels
    import numpy as np, os

    extractor = FeatureExtractor("/path/to/gym_malware/envs/utils")
    paths = [os.path.join("samples", f) for f in os.listdir("samples")][:20]
    X = extractor.extract_batch(paths)
    y = make_balanced_labels(len(X))
    models = build_models()
    train_models(models, X, y)

    result = probe_sample("samples/target.exe", extractor, models,
                          mut_fn=lambda p: append_bytes(p, 50_000))
    print(result["score_changes"])
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator

from pemutator.core.extractor import FeatureExtractor
from pemutator.analysis.delta  import sensitivity_report


def probe_sample(
    path:      str,
    extractor: FeatureExtractor,
    models:    dict[str, BaseEstimator],
    mut_fn:    callable,
    threshold: float = 1e-4,
) -> dict:
    """
    Apply a single mutation function to one PE file and measure the effect
    on feature space and model predictions.

    Parameters
    ----------
    path : str
        Path to the original PE file.
    extractor : FeatureExtractor
        Fitted extractor instance.
    models : dict[str, BaseEstimator]
        Trained classifier dict (e.g. from build_models + train_models).
    mut_fn : callable
        ``fn(path: str) -> bytes`` — the mutation to apply.
    threshold : float
        Minimum prediction change to count as "sensitive".

    Returns
    -------
    dict with keys:
      "path"          str
      "mutation"      str  (function name or "<lambda>")
      "feat_orig"     np.ndarray
      "feat_mut"      np.ndarray
      "score_before"  dict[str, float]
      "score_after"   dict[str, float]
      "report"        dict  — full sensitivity_report output
    """
    # --- original ---
    with open(path, "rb") as fh:
        orig_bytes = fh.read()
    feat_orig = extractor.extract(orig_bytes)
    X_orig = feat_orig.reshape(1, -1)

    score_before: dict[str, float] = {
        name: float(clf.predict_proba(X_orig)[0][1])
        for name, clf in models.items()
    }

    # --- mutated ---
    mut_bytes = mut_fn(path)
    feat_mut  = extractor.extract(mut_bytes)
    X_mut     = feat_mut.reshape(1, -1)

    score_after: dict[str, float] = {
        name: float(clf.predict_proba(X_mut)[0][1])
        for name, clf in models.items()
    }

    # --- analysis ---
    report = sensitivity_report(
        feat_orig, feat_mut,
        score_before, score_after,
        threshold=threshold,
    )

    return {
        "path":         path,
        "mutation":     getattr(mut_fn, "__name__", repr(mut_fn)),
        "feat_orig":    feat_orig,
        "feat_mut":     feat_mut,
        "score_before": score_before,
        "score_after":  score_after,
        "report":       report,
    }


def probe_batch(
    paths:     list[str],
    extractor: FeatureExtractor,
    models:    dict[str, BaseEstimator],
    mut_fn:    callable,
    threshold: float = 1e-4,
    verbose:   bool  = False,
) -> list[dict]:
    """
    Run probe_sample over a list of files and return results.

    Parameters
    ----------
    paths : list[str]
    extractor, models, mut_fn, threshold : same as probe_sample
    verbose : bool
        Print a one-line summary per file if True.

    Returns
    -------
    list[dict] — one result dict per path.
    """
    results = []
    for p in paths:
        try:
            r = probe_sample(p, extractor, models, mut_fn, threshold)
            results.append(r)
            if verbose:
                changes = {
                    name: d["changed"]
                    for name, d in r["report"]["score_changes"].items()
                }
                print(f"{p} | changed: {changes}")
        except Exception as exc:
            if verbose:
                print(f"{p} | ERROR: {exc}")
    return results
