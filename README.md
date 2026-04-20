# pemutator

A research library for studying how binary-level PE mutations propagate through
feature extraction and affect ML classifier decisions.

Built from the `try*.py` experimental series, packaged for use in Jupyter notebooks.

## Package Layout

```
pemutator/
├── core/
│   ├── extractor.py   # FeatureExtractor wrapping gym-malware pefeatures
│   ├── models.py      # build_models / train_models (GBDT + RF)
│   └── mutator.py     # append_bytes, add_import, pad_header, rename_section
├── analysis/
│   ├── probe.py       # probe_sample / probe_batch (single mutation, single file)
│   ├── sweep.py       # size_sweep / mutation_sensitivity (aggregate experiments)
│   └── delta.py       # feature_delta, dominant_features, group_deltas
└── viz/
    └── plots.py       # All matplotlib plotting functions
```

## Quick Start

```python
import pemutator as pm
from pemutator.core.models import make_balanced_labels

# 1. Setup
extractor = pm.FeatureExtractor("/path/to/gym_malware/envs/utils")
paths = ["samples/a.exe", "samples/b.exe", ...]
X = extractor.extract_batch(paths)
y = make_balanced_labels(len(X))

# 2. Train
models = pm.build_models()
pm.train_models(models, X, y)

# 3. Probe one file with one mutation
result = pm.probe_sample("samples/a.exe", extractor, models,
                          mut_fn=lambda p: pm.append_bytes(p, 5000))

# 4. Sweep append size to find the prediction change point
sweep = pm.size_sweep("samples/a.exe", extractor, models)

# 5. Aggregate sensitivity across files and mutations
sens = pm.mutation_sensitivity(paths[:10], extractor, models)
```

## Key Dependencies

- `lief` (tested with 0.14+)
- `scikit-learn`
- `numpy`
- `matplotlib`
- `pefeatures` from [gym-malware](https://github.com/endgameinc/gym-malware)

## Experimental Background

The `try*.py` scripts uncovered these consistent findings:
- Predictions are **piecewise-constant** w.r.t. append size
- `feature[0]` (size) changes linearly; `feature[257]` (alignment) steps discretely
- Prediction changes are driven by a **sparse subset** of the 2350-dimensional feature space
- **Append** mutations affect scores more than import/header/section changes
- **GBDT and RF** show different sensitivity profiles on identical inputs
