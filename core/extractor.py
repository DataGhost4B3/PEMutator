"""
pemutator.core.extractor
------------------------
Thin, import-safe wrapper around the gym-malware PEFeatureExtractor.

The gym-malware utils directory must be on sys.path before importing this
module.  Pass `gym_malware_utils` to FeatureExtractor.__init__ to add it
at construction time (the standard approach used throughout the try*.py
experiments).

Example
-------
    extractor = FeatureExtractor("/path/to/gym-malware/gym_malware/envs/utils")
    feat = extractor.extract(open("sample.exe", "rb").read())
    # feat is a numpy array of shape (N_FEATURES,)
"""

import sys
import numpy as np

# Patch deprecated np.int alias (numpy >=1.20 removed it)
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

_EXTRACTOR = None  # module-level singleton cache


class FeatureExtractor:
    """
    Wraps PEFeatureExtractor from the gym-malware utils package.

    Parameters
    ----------
    gym_malware_utils : str or None
        Filesystem path to the gym_malware/envs/utils directory.
        If None, assumes the path is already on sys.path.

    Attributes
    ----------
    n_features : int
        Dimensionality of the feature vector (typically 2350).
    """

    def __init__(self, gym_malware_utils: str | None = None):
        global _EXTRACTOR

        if gym_malware_utils and gym_malware_utils not in sys.path:
            sys.path.insert(0, gym_malware_utils)

        try:
            import pefeatures  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "pefeatures could not be imported. "
                "Ensure gym_malware_utils points to the "
                "gym-malware/gym_malware/envs/utils directory "
                f"(got: {gym_malware_utils!r}).\n"
                f"Original error: {exc}"
            ) from exc

        self._impl = pefeatures.PEFeatureExtractor()
        _EXTRACTOR = self

        # Probe feature length with a dummy call
        # (we cannot know ahead of time without a real PE file)
        self.n_features: int | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, raw_bytes: bytes) -> np.ndarray:
        """
        Extract a fixed-length feature vector from raw PE bytes.

        Parameters
        ----------
        raw_bytes : bytes
            Raw contents of a PE file.

        Returns
        -------
        np.ndarray, shape (n_features,)
        """
        feat = self._impl.extract(raw_bytes)
        arr = np.array(feat, dtype=np.float64)
        if self.n_features is None:
            self.n_features = len(arr)
        return arr

    def extract_file(self, path: str) -> np.ndarray:
        """
        Convenience wrapper: open a file by path and extract features.

        Parameters
        ----------
        path : str
            Path to a PE file.

        Returns
        -------
        np.ndarray, shape (n_features,)
        """
        with open(path, "rb") as fh:
            return self.extract(fh.read())

    def extract_batch(self, paths: list[str]) -> np.ndarray:
        """
        Extract features for a list of PE file paths.

        Parameters
        ----------
        paths : list[str]
            List of PE file paths.

        Returns
        -------
        np.ndarray, shape (len(paths), n_features)
            Rows where extraction failed are filled with NaN.
        """
        rows = []
        for p in paths:
            try:
                rows.append(self.extract_file(p))
            except Exception:
                rows.append(np.full(self.n_features or 2350, np.nan))
        return np.array(rows)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"FeatureExtractor(n_features={self.n_features})"
