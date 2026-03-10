# shap_selection

Feature selection for linear models using SHAP values, based on:

> Marcilio-Jr & Eler, *"From explanations to feature selection: assessing SHAP values as feature selection mechanism"*, SIBGRAPI 2020.

---

## Installation

```bash
pip install shap scikit-learn numpy
```

---

## Regression example

The full workflow in four steps: fit, rank, select, retrain.

```python
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from shap_selection import shap_select, select_by_keep_absolute, apply_feature_selection

# 1. Load and prepare data
data = fetch_california_housing()
X, y = data.data, data.target
feature_names = list(data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 2. Fit a linear model and rank features by SHAP importance
#    LinearExplainer is selected automatically — no extra flags needed
model = LinearRegression()
model.fit(X_train, y_train)

ordered_names, importance_values = shap_select(
    model, X_train, feature_names, task='regression'
)

print("Feature ranking:")
for name, score in zip(ordered_names, importance_values):
    print(f"  {name:<20} {score:.4f}")
# Feature ranking:
#   MedInc               8.2103
#   Latitude             3.1872
#   Longitude            2.9341
#   HouseAge             1.0234
#   AveOccup             0.8812
#   AveRooms             0.5023
#   Population           0.2341
#   AveBedrms            0.1987

# 3. Select the most parsimonious feature subset using the Keep Absolute metric.
#    The sweep retrains at each subset size (10%→100%) with CV, then picks the
#    smallest subset whose score is within one CV std of the peak.
results = select_by_keep_absolute(
    lambda: LinearRegression(),
    X_train, y_train,
    feature_names, ordered_names,
    task='regression',
    cv=5,
)

print(f"\nSelected {len(results['selected_features'])} of {len(feature_names)} features:")
print(f"  {results['selected_features']}")
print(f"  (fraction: {results['selected_fraction']} — peak was at {results['best_fraction']})")
# Selected 4 of 8 features:
#   ['MedInc', 'Latitude', 'Longitude', 'HouseAge']
#   (fraction: 0.4 — peak was at 1.0)

# 4. Retrain the final model on the selected features only
X_train_final = apply_feature_selection(X_train, feature_names, results['selected_features'])
X_test_final  = apply_feature_selection(X_test,  feature_names, results['selected_features'])

final_model = LinearRegression()
final_model.fit(X_train_final, y_train)

mse_full     = mean_squared_error(y_test, model.predict(X_test))
mse_selected = mean_squared_error(y_test, final_model.predict(X_test_final))
print(f"\nMSE — full model ({len(feature_names)} features): {mse_full:.4f}")
print(f"MSE — selected  ({len(results['selected_features'])} features): {mse_selected:.4f}")
# MSE — full model (8 features): 0.5561
# MSE — selected  (4 features):  0.5712
```

---

## How `select_by_keep_absolute` works

It implements the **Keep Absolute** metric from the paper (Fig. 3). For each
subset size from 10% to 100%, the model is retrained with only the top-d
SHAP-ranked features, then scored with cross-validation using R² (default).
The selected subset is the **smallest** fraction whose mean CV score is within
one standard deviation of the peak — favouring parsimony without sacrificing
meaningful performance.

R² is used as the default scorer for regression because it is positive,
bounded (0–1 for a reasonable model), and directly interpretable. The AUC
is therefore also positive and easy to compare across feature subsets.

---

## API reference

| Function | Description |
|---|---|
| `shap_select` | Rank features by SHAP importance (auto-selects Linear / Tree / Kernel explainer) |
| `select_by_keep_absolute` | Sweep subset sizes and return the most parsimonious feature set (1-std rule) |
| `select_by_knee_detection` | Sweep subset sizes and return the feature set at the point of diminishing returns |
| `keep_absolute` | Run the Keep Absolute sweep and return the full score curve and AUC |
| `shap_threshold_select` | Select features by fixed importance threshold or top-k count |
| `apply_feature_selection` | Filter a dataset to the selected feature columns |
| `KNEE_METHODS` | List of available knee detection method names |

---

## Knee detection methods

`select_by_knee_detection` supports five algorithms from [Kneeliverse](https://github.com/mariolpantunes/knee) via the `method=` argument:

| Method | Algorithm | Best for |
|---|---|---|
| `'kneedle'` (default) | Kneedle (Satopaa et al., 2011) — normalized difference signal | General use; handles multi-knee curves |
| `'dfdt'` | Dynamic First Derivative Thresholding — gradient threshold | Curves with a clear gradient change |
| `'curvature'` | Discrete curvature — maximum local curvature | Smooth curves |
| `'menger'` | Menger curvature — circumradius of point triplets | Smooth curves, alternative to curvature |
| `'lmethod'` | L-method — two-segment linear fit intersection | Piecewise-linear curves |

```python
from shap_selection import select_by_knee_detection, KNEE_METHODS

print(KNEE_METHODS)
# ['kneedle', 'dfdt', 'curvature', 'menger', 'lmethod']

# Default (kneedle)
results = select_by_knee_detection(
    lambda: LinearRegression(fit_intercept=False),
    X_train, y_train, feature_names, ordered_names,
    task='regression',
)

# Use a different algorithm
results = select_by_knee_detection(
    lambda: LinearRegression(fit_intercept=False),
    X_train, y_train, feature_names, ordered_names,
    task='regression',
    method='lmethod',
)

print(results['method'])           # 'lmethod'
print(results['knee_fraction'])    # e.g. 0.4
print(results['selected_features'])
```

Install kneeliverse to use knee detection:
```bash
pip install kneeliverse
# or, using the package extra:
pip install shap-selection[knee]
```

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
  version = {0.2.0}
}
```

The Kneedle algorithm used in `select_by_knee_detection` is one of several algorithms from [Kneeliverse](https://github.com/mariolpantunes/knee):

```bibtex
@article{antunes2025kneeliverse,
  title   = {Kneeliverse: A universal knee-detection library for performance curves},
  author  = {Antunes, Mário and Estro, Tyler},
  journal = {SoftwareX},
  year    = {2025},
  doi     = {10.1016/j.softx.2025.102089}
}
```

