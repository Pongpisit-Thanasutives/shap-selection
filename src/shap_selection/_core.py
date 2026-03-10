"""
shap_selection._core

Feature selection via SHAP values, following:
  Marcilio-Jr & Eler, "From explanations to feature selection:
  assessing SHAP values as feature selection mechanism", SIBGRAPI 2020.

Core paper methodology
----------------------
1. Compute SHAP values on the training set with the appropriate explainer:
   LinearExplainer for sklearn linear models (exact, fast),
   TreeExplainer for tree-based models (exact, fast),
   KernelExplainer as a model-agnostic fallback (approximate, slow).
2. For classification: per-class mean-absolute SHAP vectors are summed across
   classes to obtain a single importance score per feature (Fig. 2).
   For regression: mean absolute SHAP value across samples per feature.
3. Features are ranked in descending order of importance.
4. Selection quality is evaluated with the Keep Absolute metric (Fig. 3):
   the model is retrained with only the top-d features at each fraction d,
   scored with cross-validation, and the resulting curve is used to select
   the optimal feature subset automatically.

Two automatic selection strategies are provided:
  - select_by_keep_absolute  : 1-std rule (smallest subset statistically
                                as good as the best)
  - select_by_knee_detection : point of diminishing returns on the score
                                curve, via one of several Kneeliverse algorithms
"""

import warnings
import numpy as np
import shap


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shap_ordering(feature_names, shap_values, task='classification'):
    """
    Rank features by their aggregated absolute SHAP importance.

    Classification (shap_values is a list of per-class arrays):
        importance_j = sum_c  mean_i |shap_values[c][i, j]|
    Regression (shap_values is a single 2-D array):
        importance_j = mean_i |shap_values[i, j]|

    :param feature_names: array-like, length n_features
    :param shap_values:   list of (n_samples, n_features) arrays  -- classification
                          or single (n_samples, n_features) array -- regression
    :param task:          'classification' or 'regression'
    :return: (ordered_feature_names, ordered_importance_values) descending
    """
    feature_names = np.array(feature_names)

    if task == 'classification' and isinstance(shap_values, list):
        aggregated = np.sum(np.mean(np.abs(shap_values), axis=1), axis=0)
    else:
        aggregated = np.mean(np.abs(shap_values), axis=0)

    order = np.argsort(aggregated)[::-1]
    return feature_names[order], aggregated[order]


def _select_columns(X, all_names, keep_names):
    """
    Return a copy of X containing only the columns in keep_names.

    Column dropping is used rather than the paper's mean-masking strategy
    because masking creates constant (zero-variance) columns that make the
    design matrix rank-deficient for linear models, causing NaN CV scores
    (especially with fit_intercept=False).

    :param X:          numpy array (n_samples, n_features)
    :param all_names:  list of all feature names
    :param keep_names: ordered list of names to keep
    :return: numpy array (n_samples, len(keep_names))
    """
    indices = [all_names.index(name) for name in keep_names]
    return np.array(X)[:, indices]


# ---------------------------------------------------------------------------
# Explainer selection
# ---------------------------------------------------------------------------

def _is_linear_model(model):
    """Return True if the model is a scikit-learn linear model."""
    module = type(model).__module__ or ''
    return module.startswith('sklearn.linear_model')


def _build_explainer(model, X_train, background_size=0.1):
    """
    Choose and instantiate the most appropriate SHAP explainer for model.

    Priority:
      1. LinearExplainer  -- for sklearn linear models (exact, fast)
      2. TreeExplainer    -- for tree-based models (exact, fast)
      3. KernelExplainer  -- model-agnostic fallback (approximate, slow)

    :param model:           fitted model
    :param X_train:         training data; used as masker/background
    :param background_size: fraction of X_train for KernelExplainer background
    :return: (explainer, use_new_api)
             use_new_api=True  -> call explainer(X).values
             use_new_api=False -> call explainer.shap_values(X)
    """
    if _is_linear_model(model):
        masker = shap.maskers.Independent(X_train)
        explainer = shap.explainers.Linear(model, masker)
        return explainer, True

    try:
        explainer = shap.TreeExplainer(model)
        return explainer, False
    except Exception:
        pass

    background = (
        X_train if len(X_train) < 500
        else shap.sample(X_train, int(len(X_train) * background_size))
    )
    explainer = shap.KernelExplainer(model.predict, background)
    return explainer, False


# ---------------------------------------------------------------------------
# Core SHAP ordering
# ---------------------------------------------------------------------------

def shap_select(model, X_train, feature_names,
                task='classification', background_size=0.1):
    """
    Rank features by SHAP-based importance.

    The explainer is chosen automatically based on the model type:
      - sklearn linear models  -> shap.explainers.Linear  (exact, fast)
      - tree-based models      -> shap.TreeExplainer      (exact, fast)
      - everything else        -> shap.KernelExplainer    (approximate, slow)

    :param model:           fitted model
    :param X_train:         training data (used as background/masker and explained)
    :param feature_names:   array-like of feature names, length n_features
    :param task:            'classification' or 'regression'
    :param background_size: fraction of X_train for KernelExplainer background
    :return: (ordered_feature_names, importance_values) -- descending order
    """
    explainer, use_new_api = _build_explainer(model, X_train, background_size)

    if use_new_api:
        shap_values = explainer(X_train).values
    else:
        shap_values = explainer.shap_values(X_train)

    return _shap_ordering(feature_names, shap_values, task)


# ---------------------------------------------------------------------------
# Keep Absolute metric  (paper Section IV, Fig. 3)
# ---------------------------------------------------------------------------

def keep_absolute(model_factory, X, y, feature_names, ordered_feature_names,
                  task='classification', steps=None, cv=None, scoring=None):
    """
    Evaluate a SHAP feature ranking with the Keep Absolute metric.

    For each subset size d (as a fraction of total features), the model is
    retrained using only the top-d features, scored with cross-validation.
    The AUC over those scores summarises the overall quality of the ranking.

    :param model_factory:         callable returning a fresh unfitted model,
                                  e.g. ``lambda: LinearRegression()``
    :param X:                     numpy array (n_samples, n_features)
    :param y:                     target array (n_samples,)
    :param feature_names:         list of all feature names, length n_features
    :param ordered_feature_names: feature names sorted by descending importance
                                  (first output of shap_select)
    :param task:                  'classification' or 'regression'
    :param steps:                 list of fractions in (0, 1] to evaluate;
                                  defaults to [0.1, 0.2, ..., 1.0]
    :param cv:                    number of cross-validation folds (default 3)
    :param scoring:               sklearn scorer string or callable.
                                  Defaults to 'f1_weighted' (classification)
                                  or 'r2' (regression). Use higher-is-better
                                  scorers so that AUC and peak detection work.
    :return: dict with keys
             'fractions'  -- list of fractions evaluated
             'scores'     -- mean CV score at each fraction (numpy array)
             'std'        -- std of CV scores at each fraction (numpy array)
             'auc'        -- scalar AUC (trapezoidal) over fractions vs scores
    """
    from sklearn.model_selection import cross_val_score

    feature_names = list(feature_names)
    ordered_feature_names = list(ordered_feature_names)
    n_features = len(feature_names)

    unknown = set(ordered_feature_names) - set(feature_names)
    if unknown:
        raise ValueError(
            f"ordered_feature_names contains names not found in feature_names: {unknown}. "
            "Make sure shap_select was called on the same dataset passed to keep_absolute."
        )

    if steps is None:
        steps = [round(i / 10, 1) for i in range(1, 11)]
    if cv is None:
        cv = 3
    if scoring is None:
        scoring = 'f1_weighted' if task == 'classification' else 'r2'

    mean_scores, std_scores = [], []

    for frac in steps:
        n_keep = max(1, int(np.ceil(frac * n_features)))
        keep_names = ordered_feature_names[:n_keep]
        X_subset = _select_columns(np.array(X), feature_names, keep_names)

        cv_scores = cross_val_score(model_factory(), X_subset, y,
                                    cv=cv, scoring=scoring)

        if np.any(np.isnan(cv_scores)):
            raise ValueError(
                f"CV scoring produced NaN at fraction {frac} "
                f"({n_keep} features: {keep_names}). "
                "This usually means the model failed to fit — check for "
                "rank-deficient subsets, scaling issues, or a mismatch "
                "between fit_intercept and your data."
            )

        mean_scores.append(cv_scores.mean())
        std_scores.append(cv_scores.std())

    mean_scores = np.array(mean_scores)
    std_scores = np.array(std_scores)
    auc = float(np.trapz(mean_scores, x=steps))

    return {
        'fractions': steps,
        'scores':    mean_scores,
        'std':       std_scores,
        'auc':       auc,
    }


def select_by_keep_absolute(model_factory, X, y, feature_names,
                             ordered_feature_names, task='classification',
                             steps=None, cv=None, scoring=None):
    """
    Use the Keep Absolute metric to select the smallest feature subset that
    performs as well as the full model within the noise of cross-validation
    (the 1-std rule).

    Workflow:
      1. Fit a model on all features and rank them with shap_select().
      2. Call this function. It sweeps from 10% to 100% feature fractions,
         scoring each subset with cross-validation.
      3. The peak CV score is found. The smallest subset whose mean score is
         within one CV standard deviation of the peak is selected.
      4. Retrain your final model on the returned selected_features.

    :param model_factory:         callable returning a fresh unfitted model
    :param X:                     numpy array (n_samples, n_features)
    :param y:                     target array
    :param feature_names:         list of all feature names
    :param ordered_feature_names: importance-ordered feature names (from shap_select)
    :param task:                  'classification' or 'regression'
    :param steps:                 fractions to evaluate (default 0.1 .. 1.0)
    :param cv:                    CV folds (default 3)
    :param scoring:               sklearn scorer (default f1_weighted / r2)
    :return: dict with all keep_absolute outputs plus:
             'best_fraction'     -- fraction at peak CV score
             'selected_fraction' -- smallest fraction within 1 CV std of peak
             'selected_features' -- list of selected feature names
    """
    ka = keep_absolute(model_factory, X, y, feature_names,
                       ordered_feature_names, task=task,
                       steps=steps, cv=cv, scoring=scoring)

    scores = ka['scores']
    stds = ka['std']
    fractions = ka['fractions']
    ordered_feature_names = list(ordered_feature_names)

    best_idx = int(np.argmax(scores))
    tolerance = scores[best_idx] - stds[best_idx]

    selected_idx = best_idx
    for i in range(best_idx + 1):
        if scores[i] >= tolerance:
            selected_idx = i
            break

    selected_frac = fractions[selected_idx]
    n_keep = max(1, int(np.ceil(selected_frac * len(ordered_feature_names))))
    selected_features = ordered_feature_names[:n_keep]

    ka['best_fraction'] = fractions[best_idx]
    ka['selected_fraction'] = selected_frac
    ka['selected_features'] = selected_features
    return ka


# ---------------------------------------------------------------------------
# Knee detection — Kneeliverse integration
# ---------------------------------------------------------------------------

# Registry: method name -> (module_path, get_knee_fn_name)
# All single-knee functions in kneeliverse have the signature:
#   fn(x: np.ndarray, y: np.ndarray) -> int   (index into x/y)
# kneedle additionally exposes:
#   knees(points: np.ndarray, p=PeakDetection) -> np.ndarray of indices
_KNEE_REGISTRY = {
    'kneedle':   ('kneeliverse.kneedle',   'knee'),
    'dfdt':      ('kneeliverse.dfdt',      'get_knee'),
    'curvature': ('kneeliverse.curvature', 'knee'),
    'menger':    ('kneeliverse.menger',    'knee'),
    'lmethod':   ('kneeliverse.lmethod',   'knee'),
}

# Exposed publicly so users can see valid choices
KNEE_METHODS = list(_KNEE_REGISTRY.keys())


def _run_knee_method(fractions, scores, method, **method_kwargs):
    """
    Dispatch a kneeliverse knee-detection algorithm on the score curve.

    Returns the detected knee fraction as a float, or None if no knee found.

    For kneedle, multi-knee detection is used by default (PeakDetection.All).
    The last (rightmost) knee is returned, which represents the final
    significant plateau transition before the curve fully flattens — the
    most useful cut-point for feature selection.

    All other methods return a single index directly.

    :param fractions: 1-D numpy array of feature fractions (x axis)
    :param scores:    1-D numpy array of CV scores (y axis)
    :param method:    one of KNEE_METHODS
    :param method_kwargs: forwarded to the kneeliverse function
    :return: float knee fraction, or None
    """
    import importlib

    if method not in _KNEE_REGISTRY:
        raise ValueError(
            f"Unknown knee method '{method}'. Choose from: {KNEE_METHODS}"
        )

    module_path, fn_name = _KNEE_REGISTRY[method]

    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        raise ImportError(
            f"select_by_knee_detection with method='{method}' requires "
            "the 'kneeliverse' package. Install it with: pip install kneeliverse"
        )

    x = np.asarray(fractions, dtype=float)
    y = np.asarray(scores, dtype=float)

    if method == 'kneedle':
        # Use multi-knee detection by default; user can pass p=PeakDetection.Left
        # etc. via method_kwargs to override.
        from kneeliverse.kneedle import PeakDetection
        p = method_kwargs.get('p', PeakDetection.All)
        points = np.column_stack([x, y])
        indices = mod.knees(points, p=p)
        if len(indices) == 0:
            return None
        # Last knee = rightmost plateau transition
        return float(x[indices[-1]])

    # All other methods: single-knee, returns an index
    fn = getattr(mod, fn_name)
    idx = fn(x, y)

    if idx is None:
        return None
    # kneeliverse returns -1 or a negative value when no knee is found
    if isinstance(idx, (int, np.integer)) and idx < 0:
        return None

    return float(x[int(idx)])


def select_by_knee_detection(model_factory, X, y, feature_names,
                              ordered_feature_names, task='classification',
                              steps=None, cv=None, scoring=None,
                              method='kneedle', **method_kwargs):
    """
    Use the Keep Absolute metric and a knee-detection algorithm to find the
    point of diminishing returns on the score-vs-feature-fraction curve.

    The same Keep Absolute sweep as select_by_keep_absolute is run, but
    instead of the 1-std rule, a knee-detection algorithm from the
    Kneeliverse library identifies the elbow of the performance curve —
    the fraction where adding more features yields rapidly diminishing gains.

    Requires the `kneeliverse` package:
        pip install kneeliverse

    See: Antunes et al., "Kneeliverse: A universal knee-detection library for
         performance curves", SoftwareX 2025.
         https://github.com/mariolpantunes/knee

    Available methods (method=)
    ---------------------------
    'kneedle'   (default)
        Kneedle algorithm (Satopaa et al., 2011). Finds the point of maximum
        curvature via a normalized difference signal. Uses multi-knee
        detection by default (returns the last detected knee). Best
        all-round choice for feature selection curves.
        Extra kwargs: p=PeakDetection.All (default) | Left | Right | Knee

    'dfdt'
        Dynamic First Derivative Thresholding. Computes the first derivative
        of the curve and finds the point where it crosses a dynamically
        computed threshold separating "steep" from "flat" regions.
        Good when the curve has a clear gradient change.

    'curvature'
        Discrete curvature. Finds the point of maximum local curvature.
        Geometrically intuitive; works well on smooth curves.

    'menger'
        Menger curvature. Estimates curvature via the circumradius of
        triangles formed by consecutive triplets of points. Similar to
        'curvature' but uses a different estimator.

    'lmethod'
        L-method. Fits two straight-line segments to the curve and returns
        their intersection point. Simple and fast; best for curves that
        are piecewise linear.

    Fallback behaviour
    ------------------
    If the selected method cannot detect a knee (flat / monotone curve),
    a UserWarning is raised and the fraction with the highest score is
    returned instead. You can then try a different method or fall back to
    select_by_keep_absolute.

    :param model_factory:         callable returning a fresh unfitted model
    :param X:                     numpy array (n_samples, n_features)
    :param y:                     target array
    :param feature_names:         list of all feature names
    :param ordered_feature_names: importance-ordered feature names (from shap_select)
    :param task:                  'classification' or 'regression'
    :param steps:                 fractions to evaluate (default 0.1 .. 1.0)
    :param cv:                    CV folds (default 3)
    :param scoring:               sklearn scorer (default f1_weighted / r2)
    :param method:                knee detection algorithm (default 'kneedle')
    :param method_kwargs:         extra kwargs forwarded to the kneeliverse call
    :return: dict with all keep_absolute outputs plus:
             'knee_fraction'     -- fraction at the detected knee
             'selected_fraction' -- same as knee_fraction (or fallback)
             'selected_features' -- list of selected feature names
             'method'            -- name of the method used
    """
    if method not in _KNEE_REGISTRY:
        raise ValueError(
            f"Unknown knee method '{method}'. Choose from: {KNEE_METHODS}"
        )

    ka = keep_absolute(model_factory, X, y, feature_names,
                       ordered_feature_names, task=task,
                       steps=steps, cv=cv, scoring=scoring)

    fractions = ka['fractions']
    scores = ka['scores']
    ordered_feature_names = list(ordered_feature_names)

    knee_frac = _run_knee_method(fractions, scores, method, **method_kwargs)

    if knee_frac is None:
        warnings.warn(
            f"Knee detection method '{method}' could not find a knee in the "
            "score curve (curve may be too flat or monotone). "
            f"Falling back to the peak-score fraction. "
            "Try a different method or use select_by_keep_absolute instead.",
            UserWarning,
            stacklevel=2,
        )
        knee_frac = float(fractions[int(np.argmax(scores))])

    n_keep = max(1, int(np.ceil(knee_frac * len(ordered_feature_names))))
    selected_features = ordered_feature_names[:n_keep]

    ka['knee_fraction'] = knee_frac
    ka['selected_fraction'] = knee_frac
    ka['selected_features'] = selected_features
    ka['method'] = method
    return ka


# ---------------------------------------------------------------------------
# Threshold-based selection helpers
# ---------------------------------------------------------------------------

def shap_threshold_select(model, X_train, feature_names,
                          task='classification', background_size=0.1,
                          threshold=None, top_k=None):
    """
    Select a feature subset by applying a simple threshold to SHAP importance.

    Convenience wrapper for scenarios where a full Keep Absolute sweep is not
    needed. Provide at most one of threshold or top_k.

    :param model:           fitted model
    :param X_train:         training data
    :param feature_names:   array-like of feature names
    :param task:            'classification' or 'regression'
    :param background_size: fraction of X_train for KernelExplainer background
    :param threshold:       keep features with importance >= this value
    :param top_k:           keep the top-k most important features
    :return: (selected_names, selected_importance, all_ordered_names, all_ordered_importance)
    """
    if threshold is not None and top_k is not None:
        raise ValueError("Provide at most one of `threshold` or `top_k`.")

    ordered_names, ordered_importance = shap_select(
        model, X_train, feature_names,
        task=task, background_size=background_size
    )

    if threshold is not None:
        mask = ordered_importance >= threshold
    elif top_k is not None:
        top_k = min(top_k, len(ordered_names))
        mask = np.zeros(len(ordered_names), dtype=bool)
        mask[:top_k] = True
    else:
        mask = np.ones(len(ordered_names), dtype=bool)

    return (
        ordered_names[mask],
        ordered_importance[mask],
        ordered_names,
        ordered_importance,
    )


# ---------------------------------------------------------------------------
# Dataset filtering utility
# ---------------------------------------------------------------------------

def apply_feature_selection(X, all_feature_names, selected_feature_names):
    """
    Filter a dataset to retain only the selected feature columns.

    Supports both numpy arrays and pandas DataFrames.

    :param X:                      (n_samples, n_features) array or DataFrame
    :param all_feature_names:      ordered list of all feature names in X
    :param selected_feature_names: list of feature names to keep
    :return: filtered array or DataFrame with only the selected columns
    """
    all_feature_names = list(all_feature_names)
    selected_feature_names = list(selected_feature_names)

    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X[selected_feature_names]
    except ImportError:
        pass

    try:
        indices = [all_feature_names.index(name) for name in selected_feature_names]
    except ValueError as e:
        raise ValueError(
            f"Feature not found in all_feature_names: {e}. "
            "Check that selected_feature_names is a subset of all_feature_names."
        ) from e

    return np.array(X)[:, indices]
