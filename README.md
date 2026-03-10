# shap-selection

**Version 0.3.0**

Feature selection for machine learning models using SHAP values, based on:

> Marcilio-Jr & Eler, *"From explanations to feature selection: assessing SHAP values as feature selection mechanism"*, SIBGRAPI 2020.

---

## Installation

Clone the repository and install locally:

```bash
git clone https://github.com/your-username/shap-selection.git
cd shap-selection
pip install .
```

For knee detection support, also install the optional dependency:

```bash
pip install ".[knee]"
# or separately:
pip install kneeliverse
```

**Dependencies** (installed automatically): `numpy >= 1.21`, `shap >= 0.42`, `scikit-learn >= 1.0`

---

## How it works

Features are ranked by their mean absolute SHAP value. A performance curve is then built by retraining the model on increasing subsets of features (top-1, top-2, ... or top-10%, top-20%, ...) and scoring each with cross-validation. Two strategies are provided to automatically select the optimal subset from this curve:

- **`select_by_keep_absolute`** — picks the smallest subset whose CV score is within one standard deviation of the peak (*1-std rule*).
- **`select_by_knee_detection`** — finds the point of diminishing returns on the curve using one of five algorithms from [Kneeliverse](https://github.com/mariolpantunes/knee). Uses per-feature stepping by default for maximum resolution.

---

## Quick start

```python
from sklearn.linear_model import LinearRegression
from shap_selection import shap_select, select_by_keep_absolute, apply_feature_selection

# 1. Fit and rank
model = LinearRegression(fit_intercept=False)
model.fit(X_train, y_train)

ordered_names, importance_values = shap_select(
    model, X_train, feature_names, task='regression'
)

# 2. Select
results = select_by_keep_absolute(
    lambda: LinearRegression(fit_intercept=False),
    X_train, y_train, feature_names, ordered_names,
    task='regression',
)
print(results['selected_features'])  # ['feat_a', 'feat_b', ...]

# 3. Retrain on selected features
X_final = apply_feature_selection(X_train, feature_names, results['selected_features'])
final_model = LinearRegression(fit_intercept=False)
final_model.fit(X_final, y_train)
```

---

## Knee detection

`select_by_knee_detection` defaults to `step_by='feature'`, which evaluates every feature count from 1 to n individually. This gives the knee algorithm full resolution — important when features are correlated or polynomial, where coarser fraction-based stepping can skip the true elbow.

```python
from shap_selection import select_by_knee_detection, KNEE_METHODS

print(KNEE_METHODS)
# ['kneedle', 'dfdt', 'curvature', 'menger', 'lmethod']

results = select_by_knee_detection(
    lambda: LinearRegression(fit_intercept=False),
    X_train, y_train, feature_names, ordered_names,
    task='regression',
    method='kneedle',   # default
    step_by='feature',  # default: one CV fit per feature count
)

print(results['selected_features'])
print(results['knee_fraction'])   # e.g. 0.14 (7 of 49 features)
print(results['curve_shape'])     # 'concave' or 'convex' (auto-detected)
```

### Curve shape auto-detection

The library automatically infers whether the score curve is **concave** (diminishing returns — typical) or **convex** (accelerating gains — common with polynomial features). Knee algorithms are designed for concave curves; for convex curves the library bypasses the algorithm and returns the start of the final performance plateau directly. The detected shape is available in `results['curve_shape']` and `results['curve_direction']`.

### Available methods

| `method=` | Algorithm | Best for |
|---|---|---|
| `'kneedle'` (default) | Kneedle (Satopaa et al., 2011) | General use; handles multi-knee curves |
| `'dfdt'` | Dynamic First Derivative Thresholding | Curves with a clear gradient change |
| `'curvature'` | Discrete curvature | Smooth curves |
| `'menger'` | Menger curvature | Smooth curves, alternative estimator |
| `'lmethod'` | L-method (two-segment linear fit) | Piecewise-linear curves |

### `step_by` parameter

Both selection functions accept `step_by`:

| `step_by=` | Steps | CV fits (n=49 features) |
|---|---|---|
| `'fraction'` | 10%, 20%, ..., 100% | 10 |
| `'feature'` | 1, 2, ..., n features | 49 |

`select_by_knee_detection` defaults to `'feature'`; `select_by_keep_absolute` defaults to `'fraction'` (the paper's original setting). You can pass `steps=` explicitly to override both.

---

## API reference

| Function / Constant | Description |
|---|---|
| `shap_select` | Rank features by SHAP importance (auto-selects Linear / Tree / Kernel explainer) |
| `select_by_keep_absolute` | 1-std rule: smallest subset within noise of the CV peak |
| `select_by_knee_detection` | Knee detection: point of diminishing returns on the score curve |
| `keep_absolute` | Run the full sweep and return the score curve and AUC |
| `shap_threshold_select` | Select by fixed importance threshold or top-k count |
| `apply_feature_selection` | Filter a dataset to the selected feature columns |
| `KNEE_METHODS` | List of available knee detection method names |

---

## Citation

If you use this library in your research, please cite both the original paper and this repository.

**Original paper:**

```bibtex
@inproceedings{marcilio2020explanations,
  title     = {From explanations to feature selection: assessing SHAP values as feature selection mechanism},
  author    = {Marcilio-Jr, Wilson E. and Eler, Danilo M.},
  booktitle = {2020 33rd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI)},
  pages     = {340--347},
  year      = {2020},
  publisher = {IEEE},
  doi       = {10.1109/SIBGRAPI51738.2020.00053}
}
```

**This library:**

```bibtex
@software{shap_selection,
  title   = {shap-selection: Feature selection via SHAP values},
  year    = {2025},
  url     = {https://github.com/your-username/shap-selection},
  version = {0.3.0}
}
```

**Kneeliverse** (for `select_by_knee_detection`):

```bibtex
@article{antunes2025kneeliverse,
  title   = {Kneeliverse: A universal knee-detection library for performance curves},
  author  = {Antunes, Mário and Estro, Tyler},
  journal = {SoftwareX},
  year    = {2025},
  doi     = {10.1016/j.softx.2025.102089}
}
```
