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
from pemutator.analysis.evasion import (
    RandomEvasion,
    GreedyEvasion,
    run_evasion_campaign,
    evasion_summary,
)
from pemutator.analysis.fingerprint import (
    ModelFingerprint,
    empirical_importance,
    attack_surface,
    compare_fingerprints,
)

__version__ = "0.2.0"
__all__ = [
    # core
    "FeatureExtractor",
    "build_models", "train_models",
    "append_bytes", "add_import", "pad_header", "rename_section", "MUTATIONS",
    # analysis
    "probe_sample",
    "size_sweep", "mutation_sensitivity",
    "feature_delta", "dominant_features", "group_deltas", "FEATURE_GROUPS",
    # evasion
    "RandomEvasion", "GreedyEvasion",
    "run_evasion_campaign", "evasion_summary",
    # fingerprint
    "ModelFingerprint", "empirical_importance",
    "attack_surface", "compare_fingerprints",
]
