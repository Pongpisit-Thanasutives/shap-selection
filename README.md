# shap-selection

**Version 1.0.2**

Feature selection for machine learning models using SHAP values, based on:

> Marcilio-Jr & Eler, *"From explanations to feature selection: assessing SHAP values as feature selection mechanism"*, SIBGRAPI 2020.

---

## Installation

```bash
git clone https://github.com/pongpisit-thanasutives/shap-selection.git
cd shap-selection
pip install .
```

For knee detection support:

```bash
pip install ".[knee]"      # includes kneeliverse
# or separately:
pip install kneeliverse
```

**Core dependencies** (installed automatically): `numpy >= 1.21`, `shap >= 0.42`, `scikit-learn >= 1.0`, `scipy >= 1.7`

---

## How it works

1. **Rank**: compute SHAP values for a fitted model; features are sorted by mean absolute SHAP importance.
2. **Sweep**: retrain the model on the top-10%, top-20%, … features and score each subset.
3. **Select**: choose the optimal subset from the resulting score curve using one of three strategies.

Three selection strategies:

| Function | Strategy |
|---|---|
| `select_by_keep_absolute` | *1-std rule* — smallest subset whose score is within one CV std of the peak |
| `select_by_knee_detection` | *Knee* — point of diminishing returns on the score curve (Kneeliverse) |
| `auto_select` | Run both; return the one with the higher score |

---

## Quick start

```python
from sklearn.linear_model import Ridge
from shap_selection import shap_select, select_by_keep_absolute, apply_feature_selection

# 1. Fit a model and rank features by SHAP importance
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)

ordered_names, importance = shap_select(
    model, X_train, feature_names, task='regression'
)

# 2. Select features (default: r2, 3-fold CV, 1-std rule)
#    Pass the configured instance directly — no lambda needed
result = select_by_keep_absolute(
    Ridge(alpha=1.0),          # instance is cloned at each sweep step
    X_train, y_train, feature_names, ordered_names,
    task='regression',
)
print(result['selected_features'])

# 3. Retrain on selected features
X_final = apply_feature_selection(X_train, feature_names, result['selected_features'])
final_model = Ridge(alpha=1.0).fit(X_final, y_train)
```

---

## The `model_factory` argument

All sweep functions (`keep_absolute`, `select_by_keep_absolute`,
`select_by_knee_detection`, `auto_select`, `compute_criterion`) need to create
a fresh, unfitted model at every sweep step.  Three forms are accepted — pick
whichever is most natural:

```python
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor

# 1. Estimator class (default hyper-parameters)
select_by_keep_absolute(Ridge, X, y, feat, ordered)

# 2. Configured instance — the recommended form for custom hyper-parameters
#    The library calls sklearn.base.clone() on it at each step, so the
#    original object is never modified.
select_by_keep_absolute(Ridge(alpha=0.5), X, y, feat, ordered)

# 3. Zero-argument callable (lambda or factory function) — useful for
#    estimators whose clone behaviour you need to control explicitly
select_by_keep_absolute(lambda: GradientBoostingRegressor(n_estimators=50),
                        X, y, feat, ordered)
```

All three forms produce an independent, unfitted estimator at each step.

---

All three selection functions accept a unified `scoring=` parameter.
Two modes are supported with different statistical foundations.

### Cross-validated scorers (CV mode)

Any sklearn scorer string, callable, or `'llf'` (log-likelihood).

```python
result = select_by_keep_absolute(Ridge(), ..., scoring='r2')          # default for regression
result = select_by_keep_absolute(LogisticRegression(), ..., scoring='f1_weighted') # default for classification
result = select_by_keep_absolute(Ridge(), ..., scoring='llf')         # cross-validated log-likelihood
```

`cross_val_score` is used at each step. `std` across folds is meaningful and
the 1-std rule applies normally. `cv` controls the fold count (default 3).

`llf` (log-likelihood) is valid as a CV scorer because it carries no penalty
term — it is just ln L̂ evaluated on held-out data, identical in spirit to
`neg_log_loss` or `neg_mean_squared_error`.

### In-sample information criteria

`'bic'`, `'aic'`, `'sic'`, `'ebic'`, `'rebic'`

```python
result = select_by_keep_absolute(Ridge(), ..., scoring='bic')
result = select_by_keep_absolute(Ridge(), ..., scoring='ebic')
result = select_by_keep_absolute(Ridge(), ..., scoring=('ebic', 0.5))  # custom gamma
```

The model is fitted **once** on the full training subset at each step and the
criterion is evaluated on those same *n* observations. **`cv` is forced to
`None`** — `cross_val_score` is never called, regardless of the `cv` argument
passed.  `std` is 0 at every step; the 1-std rule reduces to argmax.

**Why in-sample?**
BIC and AIC contain the term k·ln(n), where n must be the sample size used to
*fit* the model. Evaluating on a held-out CV fold would use a different n for
the likelihood than for the penalty — theoretically invalid. See:
[stats.stackexchange.com/questions/319666](https://stats.stackexchange.com/questions/319666/aic-with-test-data-is-it-possible)

### Criterion formulas (higher = better, consistent with r2/f1)

| `scoring=` | Score returned | Notes |
|---|---|---|
| `'bic'` | `2·ln(L̂) − k·ln(n)` | −BIC |
| `'aic'` | `2·ln(L̂) − 2k` | −AIC |
| `'sic'` | same as `'bic'` | alias used in EBIC literature |
| `'ebic'` | `−(BIC + 2γ·ln C(p,k))` | Extended BIC (Chen & Chen 2008) |
| `'rebic'` | `−REBIC` | Robust EBIC; regression only (falls back to EBIC for classification) |
| `'llf'` | `ln(L̂)` | Cross-validated; no penalty |

For regression, ln(L̂) is the Gaussian MLE log-likelihood (σ² profiled out).
For classification, the sum form of log-loss is used.

**gamma** (γ ∈ [0, 1], default 1.0) controls the EBIC/REBIC penalty weight.
`gamma=0` reduces EBIC to BIC. Pass as a tuple: `scoring=('ebic', 0.5)`.

---

## Knee detection

`select_by_knee_detection` defaults to `step_by='feature'`, evaluating every
feature count from 1 to n. This gives the knee algorithm full resolution —
important for correlated or polynomial features where coarser fraction steps
can skip the true elbow.

```python
from shap_selection import select_by_knee_detection, KNEE_METHODS

print(KNEE_METHODS)  # ['kneedle', 'dfdt', 'curvature', 'menger', 'lmethod']

result = select_by_knee_detection(
    Ridge(alpha=1.0),           # instance is cloned each step
    X_train, y_train, feature_names, ordered_names,
    task='regression',
    method='kneedle',   # default
    scoring='r2',       # default scoring; or any IC name
)
print(result['selected_features'])
print(result['knee_fraction'])    # e.g. 0.29
print(result['curve_shape'])      # 'concave' or 'convex'
```

### Curve shape auto-detection

The library automatically detects whether the score curve is **concave**
(diminishing returns — typical) or **convex** (accelerating gains — common
with polynomial/sparse feature sets) by comparing its AUC to the area under
the chord between its endpoints.

- **Concave**: the chosen knee algorithm runs normally.
- **Convex**: the algorithm is bypassed; the library returns the start of the
  final performance plateau (the first point where gains stop).

This is critical for polynomial feature spaces: a convex-increasing curve means
performance is essentially zero until the key features are included, then jumps
sharply to a plateau. Knee algorithms designed for concave curves would return
the wrong inflection point.

### Score log-transform for unbounded IC curves

In-sample criteria (−BIC, −AIC, etc.) have no natural upper or lower bound —
a sweep might produce values spanning thousands of units.  Some knee algorithms
are sensitive to scale, or implicitly assume scores in a compact range.

Pass `log_transform_scores=True` to apply the transform
**log1p(scores − min(scores))** to the sweep curve *before* passing it to the
knee algorithm.  The raw IC values stored in `result['scores']` are never
modified; only the internal curve seen by the algorithm is transformed.

```python
result = select_by_knee_detection(
    Ridge(alpha=1.0),
    X_train, y_train, feature_names, ordered_names,
    task='regression',
    scoring='bic',               # unbounded IC sweep
    log_transform_scores=True,   # compress to [0, log1p(range)] before knee
)
print(result['log_transform_scores'])  # True
print(result['scores'])                # original −BIC values, untouched
```

**Why log1p(scores − min)?**

| Step | Effect |
|---|---|
| `− min(scores)` | Shifts the worst step to 0; all values ≥ 0 |
| `log1p(·)` | Compresses large positive values; maps 0 → 0 |
| Rank order preserved | Knee location remains statistically valid |

This is most useful when:
- Scores span hundreds or thousands of IC units across the sweep
- The knee algorithm returns the last fraction or behaves erratically on raw IC
- You are using `'bic'`, `'aic'`, `'ebic'`, or `'rebic'` as the scorer

`log_transform_scores` is also accepted by `auto_select`, where it is forwarded
to the knee detection leg only — the keep_absolute leg always operates on the
raw scores.

### Available knee methods

| `method=` | Algorithm | Best for |
|---|---|---|
| `'kneedle'` (default) | Kneedle (Satopaa et al., 2011) | General use; multi-knee curves |
| `'dfdt'` | Dynamic First Derivative Thresholding | Curves with a clear gradient change |
| `'curvature'` | Discrete curvature | Smooth curves |
| `'menger'` | Menger curvature | Smooth curves, alternative estimator |
| `'lmethod'` | L-method (two-segment linear fit) | Piecewise-linear curves |


---

## Auto-select

`auto_select` runs both `select_by_keep_absolute` and `select_by_knee_detection`
and returns the better result.  It has two operating modes.

### Unified mode (`criterion=None`, default)

Both legs share the same `scoring` and `cv`.  Scores are on the same scale, so
direct comparison is valid.

```python
from shap_selection import auto_select

# Both legs use r2 + 3-fold CV
result = auto_select(
    Ridge(alpha=1.0),           # instance is cloned each step
    X, y, feature_names, ordered_names,
    task='regression',
    scoring='r2',   # default for regression
    cv=5,
)
print(result['winner'])            # 'absolute' or 'knee'
print(result['selected_features'])
print(result['absolute_score'])    # r2 at absolute's selected fraction
print(result['knee_score'])        # r2 at knee's selected fraction
```

### Split mode (`criterion` is set)

`keep_absolute` uses `scoring` + `cv` (cross-validated, giving a real 1-std
rule with fold variance).  `select_by_knee_detection` uses `criterion`
(in-sample IC, `cv` forced to `None`), giving a principled IC elbow.

Because the two legs score on different scales, the winner is decided by
evaluating `criterion` in-sample on **both** selected subsets and comparing
those IC values.

```python
# keep_absolute: r2, 5-fold CV  →  meaningful 1-std rule
# knee_detection: BIC in-sample →  principled IC elbow, log-compressed
# winner decided by: BIC(absolute_selected) vs BIC(knee_selected)
result = auto_select(
    Ridge(alpha=1.0),
    X, y, feature_names, ordered_names,
    task='regression',
    scoring='r2',              # for keep_absolute
    cv=5,                      # for keep_absolute
    criterion='bic',           # for knee_detection + winner comparison
    log_transform_scores=True, # compress IC curve before knee algorithm
)
print(result['winner'])
print(result['selected_features'])
print(result['absolute_criterion_score'])  # BIC of absolute subset
print(result['knee_criterion_score'])      # BIC of knee subset
```

Pass a `(criterion, gamma)` tuple to set γ for EBIC/REBIC:

```python
result = auto_select(Ridge(alpha=1.0), ..., scoring='r2', criterion=('ebic', 0.5))
print(result['criterion_gamma'])  # 0.5
```

### Choosing a mode

| Situation | Recommendation |
|---|---|
| Quick / exploratory | Unified mode with `scoring='r2'` or `'f1_weighted'` |
| Want the 1-std rule to use real CV variance | Split mode: `scoring='r2'`, `criterion='bic'` |
| High-dimensional / polynomial features | Split mode with `criterion='ebic'` |
| Already using IC everywhere | Unified mode with `scoring='bic'` (both legs in-sample) |

---

---

## Reproducibility

Three sources of randomness can affect results.  Here is what each is, who controls it, and how:

### 1. Model's own randomness

Estimators like `RandomForestClassifier` or `GradientBoostingRegressor` have their own `random_state`. Embed it in the instance you pass — `clone` preserves it at every sweep step:

```python
from sklearn.ensemble import RandomForestClassifier

# Fully reproducible: random_state is cloned at each sweep step
result = select_by_keep_absolute(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X, y, feature_names, ordered_names,
)
```

Deterministic models (linear, ridge, …) have no randomness here.

### 2. CV fold splits

When `cv` is an integer, sklearn constructs `KFold` with `shuffle=False`, so splits are already deterministic given the row order of your data. No action needed in the common case.

If you want shuffled folds (e.g. for imbalanced data), pass a seeded splitter object explicitly:

```python
from sklearn.model_selection import StratifiedKFold

result = select_by_keep_absolute(
    LogisticRegression(),
    X, y, feature_names, ordered_names,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
)
```

### 3. KernelExplainer background sampling

This only applies to non-linear, non-tree models (SVMs, MLPs, KNN, …) when `len(X_train) >= 500`. For those models `shap_select` samples a background subset to approximate SHAP values. Use `random_state` to seed that sampling:

```python
from sklearn.svm import SVR

model = SVR().fit(X_train, y_train)

# Without random_state: background sample differs each call
ordered_names, imp = shap_select(model, X_train, feature_names, task='regression')

# With random_state: fully reproducible background selection
ordered_names, imp = shap_select(
    model, X_train, feature_names,
    task='regression',
    random_state=42,       # int → seeded numpy Generator
)
```

`shap_threshold_select` accepts the same `random_state` argument and forwards it.

**Linear and tree-based models are fully deterministic** — `random_state` is accepted but has no effect (no sampling is performed).

`random_state` accepts:

| Value | Behaviour |
|---|---|
| `None` (default) | Fresh OS-entropy seed each call — non-reproducible |
| `int` (e.g. `42`) | Seeded `numpy.random.Generator` — fully reproducible |
| `numpy.random.Generator` | Used directly — caller manages state |

---

## Step granularity

Both selection functions accept `step_by`:

| `step_by=` | Fractions evaluated | Fits per sweep (n=49 features) |
|---|---|---|
| `'fraction'` | 10%, 20%, …, 100% | 10 |
| `'feature'` | 1, 2, …, n features | 49 |

`select_by_keep_absolute` defaults to `'fraction'` (the paper's original setting).
`select_by_knee_detection` defaults to `'feature'` (full resolution for the knee algorithm).
Pass `steps=` explicitly to use a custom list of fractions.

---

## API reference

| Symbol | Description |
|---|---|
| `shap_select` | Rank features by SHAP importance (auto-selects Linear / Tree / Kernel explainer) |
| `keep_absolute` | Run the full sweep; return score curve and AUC |
| `select_by_keep_absolute` | 1-std rule: smallest subset within noise of the score peak |
| `select_by_knee_detection` | Knee detection: point of diminishing returns |
| `auto_select` | Run both; return the higher-scoring result |
| `compute_criterion` | Score a feature subset with BIC / AIC / EBIC / REBIC / LLF (in-sample) |
| `shap_threshold_select` | Select by fixed importance threshold or top-k count |
| `apply_feature_selection` | Filter a dataset to the selected feature columns |
| `KNEE_METHODS` | List of valid knee detection method names |
| `INSAMPLE_CRITERIA` | Set of criterion names that use in-sample scoring |
| `CRITERION_SCORERS` | All special names accepted by `scoring=` |

### `keep_absolute` result keys

| Key | Description |
|---|---|
| `fractions` | List of fractions evaluated |
| `scores` | Mean score at each fraction (higher = better) |
| `std` | Std of scores across CV folds; exactly 0 for in-sample criteria (no CV performed) |
| `auc` | Trapezoidal AUC over fractions vs scores |

### `select_by_keep_absolute` result keys

All `keep_absolute` keys plus:

| Key | Description |
|---|---|
| `best_fraction` | Fraction at peak score |
| `selected_fraction` | Fraction chosen by 1-std rule (≤ best_fraction) |
| `selected_features` | List of selected feature names |

### `select_by_knee_detection` result keys

All `keep_absolute` keys plus:

| Key | Description |
|---|---|
| `knee_fraction` | Fraction at the detected knee |
| `selected_fraction` | Same as `knee_fraction` (or peak fallback) |
| `selected_features` | List of selected feature names |
| `method` | Knee algorithm used |
| `curve_shape` | `'concave'` or `'convex'` |
| `curve_direction` | `'increasing'` or `'decreasing'` |
| `log_transform_scores` | `True` if log1p transform was applied before knee detection |

### `auto_select` result keys

Always present:

| Key | Description |
|---|---|
| `selected_features` | Winning feature list |
| `winner` | `'absolute'` or `'knee'` |
| `scoring` | The `scoring=` argument used for keep_absolute |
| `criterion` | The `criterion=` argument (`None` in unified mode) |
| `absolute_features` | Features from `select_by_keep_absolute` |
| `knee_features` | Features from `select_by_knee_detection` |
| `absolute_score` | Score at `absolute`'s selected fraction (keep_absolute curve) |
| `knee_score` | Score at `knee`'s selected fraction (knee curve) |
| `absolute_result` | Full result dict from `select_by_keep_absolute` |
| `knee_result` | Full result dict from `select_by_knee_detection` |

Split mode only (`criterion` is not `None`):

| Key | Description |
|---|---|
| `absolute_criterion_score` | IC value of absolute-selected subset (used for winner comparison) |
| `knee_criterion_score` | IC value of knee-selected subset (used for winner comparison) |
| `criterion_gamma` | γ used for EBIC/REBIC comparison (1.0 unless a tuple was passed) |

---

## Changelog

### 1.0.2

**Bug fixes:**

- **EBIC/REBIC overflow** — `comb(p, k, exact=True)` crashed with `OverflowError` or `TypeError` for datasets with ~100+ features. Replaced with `scipy.special.gammaln` for an overflow-safe log-binomial computation.
- **Multiclass SHAP handling** — Modern SHAP (≥ 0.42) returns a 3D ndarray `(n_samples, n_features, n_classes)` for multiclass classification. The old code only handled the legacy list-of-2D format, silently producing wrong feature rankings for 3D arrays.
- **REBIC null-model RSS** — The intercept-only residual sum of squares used `‖y‖²` instead of `‖y − ȳ‖²`, giving incorrect REBIC scores when the target had a non-zero mean.

**Improvements:**

- Narrowed the bare `except Exception` in `_build_explainer` to specific error types.
- `_select_columns` and `apply_feature_selection` now use O(1) dict lookups instead of O(n) linear scans.
- Replaced `np.array(X)` with `np.asarray(X)` throughout to avoid unnecessary copies when X is already an ndarray.
- `selected_features`, `absolute_features`, and `knee_features` in result dicts are now consistently Python lists across all functions.

---

## Citation

If you use this library in research, please cite the original paper and this repository.

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
  author  = {Thanasutives, Pongpisit},
  year    = {2025},
  url     = {https://github.com/pongpisit-thanasutives/shap-selection},
  version = {1.0.2}
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

**Extended BIC** (for `scoring='ebic'` or `'rebic'`):

```bibtex
@article{chen2008ebic,
  title   = {Extended Bayesian information criteria for model selection with large model spaces},
  author  = {Chen, Jiahua and Chen, Zehua},
  journal = {Biometrika},
  volume  = {95},
  number  = {3},
  pages   = {759--771},
  year    = {2008},
  doi     = {10.1093/biomet/asn034}
}
```
