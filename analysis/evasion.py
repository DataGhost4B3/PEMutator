"""
pemutator.analysis.evasion
--------------------------
Greedy and random evasion loops.

This module bridges the controlled perturbation experiments (try*.py) and
the full RL-based attack described in Anderson et al. (2018).  Instead of
a trained RL agent we provide two simpler baselines:

  RandomEvasion   — try mutations in random order, accept the first that
                    lowers the malicious-class score below ``target``.

  GreedyEvasion   — at each step, try every mutation and commit to whichever
                    produces the largest drop in the malicious-class score.
                    Equivalent to a hill-climbing / steepest-descent policy.

Both classes share the same interface and return a structured EvasionResult.

Key design choices (informed by the try*.py findings)
------------------------------------------------------
* We track the full score trajectory, not just the endpoint.
* At each step we also record the dominant feature index so we can later
  plot *which* feature drove each successful step.
* A step is "successful" if the score drops by more than ``step_threshold``.
* Evasion is declared when score drops below ``target`` (default 0.5).
* We cap iterations at ``max_steps`` to avoid infinite loops.

Example
-------
    from pemutator.analysis.evasion import GreedyEvasion, RandomEvasion
    from pemutator import FeatureExtractor, build_models, MUTATIONS

    evader = GreedyEvasion(extractor, models["RF"], MUTATIONS, target=0.5)
    result = evader.run("samples/target.exe")

    print(result.evaded, result.steps_taken, result.final_score)
"""

from __future__ import annotations

import random
import copy
import tempfile
import os
import numpy as np
from dataclasses import dataclass, field
from sklearn.base import BaseEstimator

from pemutator.core.extractor import FeatureExtractor
from pemutator.analysis.delta  import feature_delta, dominant_features, IMPORTANT_IDX


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """Record of a single evasion step."""
    step:          int
    mutation:      str
    score_before:  float
    score_after:   float
    score_delta:   float          # negative = score dropped (good)
    dominant_idx:  int
    feat_delta_l1: float          # L1 norm of full feature delta


@dataclass
class EvasionResult:
    """Full result of one evasion run."""
    path:         str
    model_name:   str
    strategy:     str             # "greedy" or "random"
    target:       float
    evaded:       bool
    steps_taken:  int
    initial_score: float
    final_score:  float
    trajectory:   list[float]    = field(default_factory=list)   # score per step
    steps:        list[StepRecord] = field(default_factory=list)
    dominant_history: list[int]  = field(default_factory=list)   # dominant feat per step

    @property
    def score_drop(self) -> float:
        return self.initial_score - self.final_score

    @property
    def effective_mutations(self) -> list[str]:
        """Names of mutation steps that actually lowered the score."""
        return [s.mutation for s in self.steps if s.score_delta < 0]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class _BaseEvader:
    """
    Shared plumbing for evasion strategies.

    Parameters
    ----------
    extractor   : FeatureExtractor
    model       : sklearn estimator (must implement predict_proba)
    mutations   : dict[str, callable]  —  name → fn(path) → bytes
    target      : float  — score below which we declare evasion
    max_steps   : int    — iteration cap
    step_threshold : float — minimum score drop per step to count as "useful"
    """

    def __init__(
        self,
        extractor:       FeatureExtractor,
        model:           BaseEstimator,
        mutations:       dict[str, callable],
        target:          float = 0.5,
        max_steps:       int   = 10,
        step_threshold:  float = 1e-4,
    ):
        self.extractor      = extractor
        self.model          = model
        self.mutations      = mutations
        self.target         = target
        self.max_steps      = max_steps
        self.step_threshold = step_threshold

    def _score(self, raw_bytes: bytes) -> tuple[float, np.ndarray]:
        feat  = self.extractor.extract(raw_bytes)
        X     = feat.reshape(1, -1)
        score = float(self.model.predict_proba(X)[0][1])
        return score, feat

    def _write_temp(self, data: bytes) -> str:
        """Write bytes to a temp file and return its path."""
        fd, path = tempfile.mkstemp(suffix=".exe")
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        return path

    def _apply(self, current_bytes: bytes, mut_name: str) -> bytes:
        """Apply a named mutation to raw bytes via a temp file."""
        tmp = self._write_temp(current_bytes)
        try:
            result = self.mutations[mut_name](tmp)
        finally:
            os.unlink(tmp)
        return result

    def run(self, path: str) -> EvasionResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Random evasion
# ---------------------------------------------------------------------------

class RandomEvasion(_BaseEvader):
    """
    Randomly sample mutations without replacement per step.
    At each step, shuffle mutation names and try them in order;
    commit to the first one that produces any score drop.

    This matches the "random uniform exploration policy" baseline
    in Table 2 of Anderson et al. (2018).
    """

    strategy = "random"

    def run(self, path: str) -> EvasionResult:
        with open(path, "rb") as fh:
            current_bytes = fh.read()

        initial_score, feat_orig = self._score(current_bytes)
        score    = initial_score
        traj     = [score]
        steps    = []
        dom_hist = []

        for step_i in range(self.max_steps):
            if score <= self.target:
                break

            order = list(self.mutations.keys())
            random.shuffle(order)

            committed = False
            for mut_name in order:
                try:
                    candidate = self._apply(current_bytes, mut_name)
                    new_score, feat_mut = self._score(candidate)
                except Exception:
                    continue

                delta_vec = feature_delta(feat_orig, feat_mut)
                dom       = dominant_features(delta_vec, indices=IMPORTANT_IDX)

                rec = StepRecord(
                    step          = step_i,
                    mutation      = mut_name,
                    score_before  = score,
                    score_after   = new_score,
                    score_delta   = new_score - score,
                    dominant_idx  = dom["dominant_idx"],
                    feat_delta_l1 = float(delta_vec.sum()),
                )
                steps.append(rec)

                if new_score < score - self.step_threshold:
                    current_bytes = candidate
                    feat_orig     = feat_mut
                    score         = new_score
                    traj.append(score)
                    dom_hist.append(dom["dominant_idx"])
                    committed = True
                    break

            if not committed:
                # No mutation helped — record a no-op step
                traj.append(score)

        return EvasionResult(
            path           = path,
            model_name     = getattr(self.model, "__class__", type(self.model)).__name__,
            strategy       = self.strategy,
            target         = self.target,
            evaded         = score <= self.target,
            steps_taken    = len(steps),
            initial_score  = initial_score,
            final_score    = score,
            trajectory     = traj,
            steps          = steps,
            dominant_history = dom_hist,
        )


# ---------------------------------------------------------------------------
# Greedy evasion
# ---------------------------------------------------------------------------

class GreedyEvasion(_BaseEvader):
    """
    At each step, evaluate ALL mutations and commit to whichever produces
    the largest score drop.  Ties are broken arbitrarily.

    This is a deterministic hill-climbing / steepest-descent policy.
    Unlike random evasion, it exhausts the action space per step and
    always picks the locally optimal move.
    """

    strategy = "greedy"

    def run(self, path: str) -> EvasionResult:
        with open(path, "rb") as fh:
            current_bytes = fh.read()

        initial_score, feat_orig = self._score(current_bytes)
        score    = initial_score
        traj     = [score]
        steps    = []
        dom_hist = []

        for step_i in range(self.max_steps):
            if score <= self.target:
                break

            best_score    = score
            best_bytes    = None
            best_feat     = None
            best_mut_name = None
            best_dom      = None
            best_l1       = 0.0

            for mut_name in self.mutations:
                try:
                    candidate = self._apply(current_bytes, mut_name)
                    new_score, feat_mut = self._score(candidate)
                except Exception:
                    continue

                if new_score < best_score:
                    delta_vec  = feature_delta(feat_orig, feat_mut)
                    dom        = dominant_features(delta_vec, indices=IMPORTANT_IDX)
                    best_score    = new_score
                    best_bytes    = candidate
                    best_feat     = feat_mut
                    best_mut_name = mut_name
                    best_dom      = dom["dominant_idx"]
                    best_l1       = float(delta_vec.sum())

            if best_bytes is None or best_score >= score - self.step_threshold:
                # No improvement found
                traj.append(score)
                break

            rec = StepRecord(
                step          = step_i,
                mutation      = best_mut_name,
                score_before  = score,
                score_after   = best_score,
                score_delta   = best_score - score,
                dominant_idx  = best_dom,
                feat_delta_l1 = best_l1,
            )
            steps.append(rec)

            current_bytes = best_bytes
            feat_orig     = best_feat
            score         = best_score
            traj.append(score)
            dom_hist.append(best_dom)

        return EvasionResult(
            path           = path,
            model_name     = getattr(self.model, "__class__", type(self.model)).__name__,
            strategy       = self.strategy,
            target         = self.target,
            evaded         = score <= self.target,
            steps_taken    = len(steps),
            initial_score  = initial_score,
            final_score    = score,
            trajectory     = traj,
            steps          = steps,
            dominant_history = dom_hist,
        )


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_evasion_campaign(
    paths:      list[str],
    extractor:  FeatureExtractor,
    models:     dict[str, BaseEstimator],
    mutations:  dict[str, callable],
    strategy:   str   = "greedy",
    target:     float = 0.5,
    max_steps:  int   = 10,
    verbose:    bool  = True,
) -> dict[str, list[EvasionResult]]:
    """
    Run an evasion campaign across multiple files and models.

    Parameters
    ----------
    paths      : list[str] — PE files to attack
    extractor  : FeatureExtractor
    models     : dict[str, BaseEstimator]
    mutations  : dict[str, callable]
    strategy   : "greedy" or "random"
    target     : evasion threshold (score below which = evaded)
    max_steps  : per-file step cap
    verbose    : print progress

    Returns
    -------
    dict[model_name, list[EvasionResult]]
    """
    EvaderCls = GreedyEvasion if strategy == "greedy" else RandomEvasion

    results: dict[str, list[EvasionResult]] = {name: [] for name in models}

    for name, clf in models.items():
        evader = EvaderCls(extractor, clf, mutations, target=target, max_steps=max_steps)
        if verbose:
            print(f"\n── {strategy.upper()} evasion against {name} ──")

        for path in paths:
            try:
                r = evader.run(path)
                results[name].append(r)
                if verbose:
                    status = "EVADED ✓" if r.evaded else "failed  "
                    print(
                        f"  {status}  {os.path.basename(path):<30}  "
                        f"{r.initial_score:.3f} → {r.final_score:.3f}  "
                        f"({r.steps_taken} steps)"
                    )
            except Exception as exc:
                if verbose:
                    print(f"  ERROR   {os.path.basename(path)}: {exc}")

    return results


def evasion_summary(results: dict[str, list[EvasionResult]]) -> dict:
    """
    Compute aggregate statistics from a campaign result.

    Returns
    -------
    dict[model_name, dict] with keys:
      n_files, n_evaded, evasion_rate, avg_steps, avg_score_drop,
      mutation_counts (how often each mutation contributed)
    """
    summary = {}
    for name, res_list in results.items():
        n_evaded     = sum(r.evaded for r in res_list)
        n_files      = len(res_list)
        avg_steps    = np.mean([r.steps_taken for r in res_list]) if res_list else 0
        avg_drop     = np.mean([r.score_drop for r in res_list])  if res_list else 0

        mut_counts: dict[str, int] = {}
        for r in res_list:
            for m in r.effective_mutations:
                mut_counts[m] = mut_counts.get(m, 0) + 1

        summary[name] = {
            "n_files":       n_files,
            "n_evaded":      n_evaded,
            "evasion_rate":  n_evaded / max(n_files, 1),
            "avg_steps":     float(avg_steps),
            "avg_score_drop": float(avg_drop),
            "mutation_counts": mut_counts,
        }
    return summary
