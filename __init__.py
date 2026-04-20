"""
pemutator
=========
A research library for studying how binary-level PE mutations
propagate through feature extraction and affect ML classifier decisions.

Built on top of the gym-malware / EMBER feature pipeline and LIEF.

Modules
-------
pemutator.core.extractor   - Feature extraction wrapper (PEFeatureExtractor)
pemutator.core.models      - Classifier factory and training helpers
pemutator.core.mutator     - LIEF-backed PE mutation functions
pemutator.analysis.probe   - Single-sample probing utilities
pemutator.analysis.sweep   - Size-sweep and multi-mutation sensitivity sweeps
pemutator.analysis.delta   - Feature-delta and dominant-index analysis
pemutator.viz.plots        - Matplotlib plotting helpers
"""

from pemutator.core.extractor import FeatureExtractor
from pemutator.core.models    import build_models, train_models
from pemutator.core.mutator   import (
    append_bytes,
    add_import,
    pad_header,
    rename_section,
    MUTATIONS,
)
from pemutator.analysis.probe  import probe_sample
from pemutator.analysis.sweep  import size_sweep, mutation_sensitivity
from pemutator.analysis.delta  import (
    feature_delta,
    dominant_features,
    group_deltas,
    FEATURE_GROUPS,
)

__version__ = "0.1.0"
__all__ = [
    "FeatureExtractor",
    "build_models", "train_models",
    "append_bytes", "add_import", "pad_header", "rename_section", "MUTATIONS",
    "probe_sample",
    "size_sweep", "mutation_sensitivity",
    "feature_delta", "dominant_features", "group_deltas", "FEATURE_GROUPS",
]
