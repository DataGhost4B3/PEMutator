"""
pemutator.core.models
---------------------
Classifier factory and training utilities.

The experiments used two scikit-learn models:
  * GradientBoostingClassifier  (modelA / GBDT)
  * RandomForestClassifier      (modelB / RF)

This module provides helpers to build, train and score them in a
consistent way.  It intentionally stays lightweight so the researcher
can swap in any sklearn-compatible estimator.

Example
-------
    models = build_models()
    train_models(models, X, y)
    score = models["GBDT"].predict_proba(X_test)[:, 1]
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.base import BaseEstimator
from typing import Any


# ---------------------------------------------------------------------------
# Defaults (mirroring the try*.py experiments)
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, dict[str, Any]] = {
    "GBDT": dict(n_estimators=50, random_state=42),
    "RF":   dict(n_estimators=50, random_state=42),
}


def build_models(
    gbdt_kwargs: dict | None = None,
    rf_kwargs:   dict | None = None,
) -> dict[str, BaseEstimator]:
    """
    Instantiate (but do not train) the two baseline classifiers.

    Parameters
    ----------
    gbdt_kwargs : dict, optional
        Override keyword arguments for GradientBoostingClassifier.
    rf_kwargs : dict, optional
        Override keyword arguments for RandomForestClassifier.

    Returns
    -------
    dict with keys "GBDT" and "RF", each holding an untrained estimator.
    """
    gbdt_kw = {**_DEFAULTS["GBDT"], **(gbdt_kwargs or {})}
    rf_kw   = {**_DEFAULTS["RF"],   **(rf_kwargs   or {})}

    return {
        "GBDT": GradientBoostingClassifier(**gbdt_kw),
        "RF":   RandomForestClassifier(**rf_kw),
    }


def train_models(
    models: dict[str, BaseEstimator],
    X: np.ndarray,
    y: np.ndarray | list,
) -> dict[str, BaseEstimator]:
    """
    Fit all models in ``models`` on (X, y) and return them.

    Parameters
    ----------
    models : dict[str, BaseEstimator]
        Dict of name → estimator (as returned by build_models).
    X : np.ndarray, shape (n_samples, n_features)
    y : array-like, shape (n_samples,)
        Binary labels (0 = benign, 1 = malicious).

    Returns
    -------
    The same dict, with each estimator fitted in place.
    """
    y = np.array(y)
    for name, clf in models.items():
        clf.fit(X, y)
    return models


def make_balanced_labels(n: int) -> list[int]:
    """
    Generate a balanced binary label vector of length n.
    Used as a stand-in when real labels are unavailable (as in the
    try*.py experiments which used synthetic/fake labels).

    Parameters
    ----------
    n : int
        Total number of samples.

    Returns
    -------
    list[int] with n//2 zeros followed by n - n//2 ones.
    """
    return [0] * (n // 2) + [1] * (n - n // 2)


def predict_proba_malicious(
    models: dict[str, BaseEstimator],
    X: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Return the probability of the malicious class (index 1) for each model.

    Parameters
    ----------
    models : dict[str, BaseEstimator]
    X : np.ndarray, shape (1, n_features) or (n_samples, n_features)

    Returns
    -------
    dict[str, np.ndarray] — shape (n_samples,) per model.
    """
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return {
        name: clf.predict_proba(X)[:, 1]
        for name, clf in models.items()
    }
