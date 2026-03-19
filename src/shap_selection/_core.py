"""
shap_selection._core
====================

Feature selection via SHAP values, based on:

  Marcilio-Jr & Eler, "From explanations to feature selection: assessing
  SHAP values as feature selection mechanism", SIBGRAPI 2020.

Algorithm
---------
1. Compute SHAP values on the training set with the appropriate explainer:
     LinearExplainer  — sklearn linear models  (exact, fast)
     TreeExplainer    — tree-based models       (exact, fast)
     KernelExplainer  — everything else         (approximate, slow)
2. For classification, per-class mean-absolute SHAP vectors are summed across
   classes to obtain a single importance score per feature (paper Fig. 2).
   For regression, mean absolute SHAP value across samples per feature.
3. Features are ranked in descending order of importance.
4. Selection quality is evaluated with the Keep Absolute metric (paper Fig. 3):
   the model is retrained with only the top-d features at each fraction d,
   then scored.  Two scoring modes are supported:

   a) Cross-validated scorers  (sklearn strings, callables, or 'llf')
      Scored via cross_val_score.  The 1-std rule — or the knee of the
      score curve — selects the final subset.

   b) In-sample information criteria  ('bic', 'aic', 'sic', 'ebic', 'rebic')
      The model is fitted ONCE on the full training set at each step and the
      criterion is evaluated on those same n observations.  This is the ONLY
      statistically valid approach: the BIC/AIC penalty term k*ln(n) must
      share the same n as the log-likelihood.  Using a held-out CV fold would
      mix test-set likelihood with a training-set penalty — theoretically
      invalid.  No cross-validation is performed; std = 0.

Model factory
-------------
All sweep functions accept *model_factory* in any of three forms:

  LinearRegression                 estimator class  (called with no args)
  Ridge(alpha=0.1)                 estimator instance  (cloned each step)
  lambda: Ridge(alpha=0.1)         zero-arg callable  (called each step)

The helper ``_coerce_factory`` normalises all three forms to a consistent
zero-arg callable before any sweep begins.

Reproducibility
---------------
Three sources of randomness exist in a typical run:

1. **Model's own randomness** (RandomForest, GBM, …)
   Controlled entirely by the user — embed ``random_state`` in the
   estimator instance passed as ``model_factory``.  ``clone`` preserves it.

2. **CV fold splits**
   When ``cv`` is an integer, sklearn constructs ``KFold``/``StratifiedKFold``
   with ``shuffle=False``, making splits deterministic given the data order.
   If you need shuffled folds, pass a seeded splitter object explicitly:
   ``cv=KFold(n_splits=5, shuffle=True, random_state=42)``.

3. **KernelExplainer background sampling**
   Only fires for non-linear, non-tree models when ``len(X_train) >= 500``.
   Controlled via the ``random_state`` parameter on ``shap_select`` and
   ``shap_threshold_select`` (``None`` / ``int`` / ``numpy.random.Generator``).
   Linear and tree-based models are fully deterministic and ignore this.

Scoring convention
------------------
All scores follow **higher = better**, consistent with sklearn's r2, f1, etc.:

  BIC   returned as  2*ln(L) - k*ln(n)          (negated BIC)
  AIC   returned as  2*ln(L) - 2k               (negated AIC)
  EBIC  returned as  -(BIC + 2*gamma*ln C(p,k))  (negated EBIC)
  REBIC returned as  -REBIC                       (negated REBIC)
  llf   returned as  ln(L)                        (already higher = better)
"""

import warnings
import numpy as np
import shap
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import cross_val_score

# NumPy >= 2.0 renamed trapz -> trapezoid
_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _shap_ordering(feature_names, shap_values, task='classification'):
    feature_names = np.array(feature_names)
    # Modern SHAP (>=0.42) returns a 3D ndarray (n_samples, n_features, n_classes)
    # for multiclass classification; older versions return a list of 2D arrays.
    # Both forms need per-class mean-absolute values summed across classes.
    arr = np.asarray(shap_values) if not isinstance(shap_values, list) \
        else np.array(shap_values)
    if task == 'classification' and arr.ndim == 3:
        # arr is (n_samples, n_features, n_classes) for modern SHAP
        # or (n_classes, n_samples, n_features) from np.array(list-of-2D)
        if isinstance(shap_values, list):
            # Legacy list: each element is (n_samples, n_features)
            # np.array -> (n_classes, n_samples, n_features)
            aggregated = np.sum(np.mean(np.abs(arr), axis=1), axis=0)
        else:
            # Modern 3D ndarray: (n_samples, n_features, n_classes)
            aggregated = np.sum(np.mean(np.abs(arr), axis=0), axis=1)
    else:
        aggregated = np.mean(np.abs(arr), axis=0)
    order = np.argsort(aggregated)[::-1]
    return feature_names[order], aggregated[order]


def _select_columns(X, all_names, keep_names):
    all_names  = list(all_names)
    keep_names = list(keep_names)
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    indices = [name_to_idx[name] for name in keep_names]
    return np.asarray(X)[:, indices]


def _log_transform_scores(scores):
    """
    Apply a relative log transform to a score array.

    Transform: log1p(scores - scores.min())

    This compresses unbounded IC curves (−BIC, −AIC, etc.) into a
    [0, log1p(range)] interval while preserving rank order and curve shape.
    It is applied only to the scores fed to the knee algorithm; the raw IC
    values stored in the result dict are always untransformed.

    Derivation
    ----------
    Subtract minimum  ->  shifts all values to [0, range]; worst step = 0.
    log1p             ->  log(1 + x), giving 0 at the minimum and
                          log1p(range) at the maximum.  Using +1 rather than
                          +eps gives a natural unit: a 1-unit IC difference
                          maps to log(2) ≈ 0.693 — comparable to typical
                          IC differences between adjacent feature subsets.

    :param scores: array-like of floats
    :return:       transformed numpy array, same shape, dtype float
    """
    s = np.asarray(scores, dtype=float)
    return np.log1p(s - s.min())


def _coerce_factory(model_factory):
    """
    Normalise *model_factory* to a zero-argument callable that returns a
    fresh, unfitted sklearn estimator on every call.

    Accepted forms
    --------------
    Estimator class
        ``LinearRegression``, ``Ridge``, …
        The class is returned unchanged; calling it with ``()`` already
        produces a fresh default instance.

    Estimator instance (fitted **or** unfitted)
        ``Ridge(alpha=0.1)``, a pre-fitted ``LinearRegression()``, …
        Wrapped with ``sklearn.base.clone`` so each sweep step gets an
        independent copy with the same hyper-parameters but no fitted state.

    Zero-argument callable
        ``lambda: Ridge(alpha=0.1)``, a factory function, …
        Returned unchanged.

    :param model_factory: estimator class, instance, or zero-arg callable
    :return:              zero-arg callable producing a fresh estimator
    :raises TypeError:    if none of the above forms match
    """
    if isinstance(model_factory, type) and issubclass(model_factory, BaseEstimator):
        # e.g. LinearRegression — calling the class gives a fresh default instance
        return model_factory
    if isinstance(model_factory, BaseEstimator):
        # e.g. Ridge(alpha=0.1) — clone preserves hyper-params, drops fitted state
        return lambda: clone(model_factory)
    if callable(model_factory):
        # lambda, function, or any other zero-arg callable
        return model_factory
    raise TypeError(
        "model_factory must be an sklearn estimator class (e.g. LinearRegression), "
        "an estimator instance (e.g. Ridge(alpha=0.1)), or a zero-argument callable "
        f"(e.g. lambda: Ridge(alpha=0.1)).  Got: {type(model_factory)!r}"
    )


# ---------------------------------------------------------------------------
# Randomness helpers
# ---------------------------------------------------------------------------

def _resolve_rng(random_state):
    """
    Normalise *random_state* to a ``numpy.random.Generator``.

    Accepted forms
    --------------
    None
        A fresh Generator seeded from OS entropy — non-reproducible but
        statistically independent across calls.
    int
        A Generator seeded with that integer — fully reproducible.
    numpy.random.Generator
        Returned unchanged — caller controls the state.

    :param random_state: None, int, or numpy.random.Generator
    :return:             numpy.random.Generator
    :raises TypeError:   for any other type
    """
    if random_state is None or isinstance(random_state, (int, np.integer)):
        return np.random.default_rng(random_state)
    if isinstance(random_state, np.random.Generator):
        return random_state
    raise TypeError(
        "random_state must be None, a non-negative integer, or a "
        f"numpy.random.Generator instance.  Got: {type(random_state)!r}"
    )


# ---------------------------------------------------------------------------
# Explainer selection
# ---------------------------------------------------------------------------

def _is_linear_model(model):
    module = type(model).__module__ or ''
    return module.startswith('sklearn.linear_model')


def _build_explainer(model, X_train, background_size=0.1, rng=None):
    """
    Build the best SHAP explainer for *model*.

    For KernelExplainer, when ``len(X_train) >= 500``, a random background
    subset of size ``background_size * len(X_train)`` is drawn using *rng*.
    Pass a seeded Generator to make the background selection reproducible.

    :param rng: numpy.random.Generator (from _resolve_rng); None means fresh
    """
    if _is_linear_model(model):
        masker    = shap.maskers.Independent(X_train)
        explainer = shap.explainers.Linear(model, masker)
        return explainer, True
    _tree_errors = [TypeError, ValueError, AttributeError, ImportError]
    if hasattr(shap.utils, 'InvalidModelError'):
        _tree_errors.append(shap.utils.InvalidModelError)
    try:
        return shap.TreeExplainer(model), False
    except tuple(_tree_errors):
        pass
    X_arr = np.array(X_train)
    if len(X_arr) < 500:
        background = X_arr
    else:
        if rng is None:
            rng = np.random.default_rng()
        n_bg       = max(1, int(len(X_arr) * background_size))
        idx        = rng.choice(len(X_arr), size=n_bg, replace=False)
        background = X_arr[idx]
    return shap.KernelExplainer(model.predict, background), False


# ---------------------------------------------------------------------------
# Public: SHAP-based feature ranking
# ---------------------------------------------------------------------------

def shap_select(model, X_train, feature_names,
                task='classification', background_size=0.1,
                random_state=None):
    """
    Rank features by SHAP-based importance.

    Explainer priority:
      sklearn linear models -> shap.explainers.Linear (exact, fast)
      tree-based models     -> shap.TreeExplainer     (exact, fast)
      everything else       -> shap.KernelExplainer   (approximate, slow)

    For KernelExplainer models only, when ``len(X_train) >= 500`` a random
    background subset is sampled.  Pass ``random_state`` to make that
    sampling reproducible.  Linear and tree-based models are fully
    deterministic and ignore ``random_state``.

    :param model:           fitted model
    :param X_train:         training data
    :param feature_names:   array-like of feature names
    :param task:            'classification' or 'regression'
    :param background_size: fraction of X_train used as KernelExplainer
                            background when len(X_train) >= 500
    :param random_state:    seed for background sampling (KernelExplainer only).
                            None  — non-reproducible (OS entropy each call)
                            int   — reproducible (e.g. random_state=42)
                            numpy.random.Generator — advanced: caller-managed
    :return: (ordered_feature_names, importance_values) descending
    """
    rng = _resolve_rng(random_state)
    explainer, use_new_api = _build_explainer(model, X_train, background_size, rng)
    shap_vals = explainer(X_train).values if use_new_api else explainer.shap_values(X_train)
    return _shap_ordering(feature_names, shap_vals, task)


# ---------------------------------------------------------------------------
# Scoring dispatch
# ---------------------------------------------------------------------------

#: In-sample information criteria.
#:
#: BIC/AIC/EBIC/REBIC require the model to be fitted and evaluated on the
#: *same* n observations.  The penalty term k*ln(n) is derived under the
#: assumption that n equals the training-set size.  Using a CV held-out fold
#: (where n_test < n_train) is statistically invalid.
#: keep_absolute bypasses cross_val_score for these; std = 0; cv is ignored.
INSAMPLE_CRITERIA = frozenset({'bic', 'aic', 'sic', 'ebic', 'rebic'})

#: llf is CV-compatible: ln(L) on held-out data carries no n-dependent
#: penalty term.  It is equivalent in spirit to neg-log-loss or neg-MSE.
_LLF_SCORER_NAME = 'llf'

#: All special names accepted by scoring=.
CRITERION_SCORERS = INSAMPLE_CRITERIA | {_LLF_SCORER_NAME}


def _make_llf_scorer(task):
    """
    sklearn-compatible scorer returning ln(L_hat) on held-out data.

    Higher is better.  No n-dependent penalty so CV folds are valid.
    """
    from sklearn.metrics import log_loss

    def scorer(estimator, X, y):
        n = len(y)
        if task == 'regression':
            y_pred = estimator.predict(X)
            rss    = max(float(np.sum((y - y_pred) ** 2)), 1e-10)
            return -n / 2.0 * np.log(2.0 * np.pi * rss / n) - n / 2.0
        else:
            proba = estimator.predict_proba(X)
            return float(-log_loss(y, proba, normalize=False))

    return scorer


def _parse_scoring(scoring, task):
    """
    Parse scoring= into (cv_scoring, insample_criterion, gamma).

    cv_scoring         — passed to cross_val_score (or None for in-sample)
    insample_criterion — criterion name (or None for CV path)
    gamma              — EBIC/REBIC weight (default 1.0)

    Rules
    -----
    None / sklearn str / callable          -> CV path
    'llf'                                  -> CV path (llf scorer)
    'bic'|'aic'|'sic'|'ebic'|'rebic'       -> in-sample path, cv ignored
    ('ebic', 0.5) tuple                    -> in-sample with custom gamma
    """
    gamma = 1.0

    if isinstance(scoring, tuple):
        if (len(scoring) != 2 or
                not isinstance(scoring[0], str) or
                scoring[0].lower() not in INSAMPLE_CRITERIA):
            raise ValueError(
                f"tuple scoring must be (criterion, gamma) where criterion in "
                f"{sorted(INSAMPLE_CRITERIA)}, got {scoring!r}"
            )
        return None, scoring[0].lower(), float(scoring[1])

    if isinstance(scoring, str):
        s = scoring.lower()
        if s == _LLF_SCORER_NAME:
            return _make_llf_scorer(task), None, gamma
        if s in INSAMPLE_CRITERIA:
            return None, s, gamma
        return scoring, None, gamma   # assume valid sklearn scorer string

    return scoring, None, gamma       # None or callable


# ---------------------------------------------------------------------------
# In-sample information criteria
# ---------------------------------------------------------------------------

def _compute_criterion(model_factory, X, y, feature_names,
                        selected_features, task, criterion, gamma=1.0):
    """
    Fit a fresh model on selected_features and compute an in-sample IC.

    All criteria are returned higher = better:

      'bic' / 'sic'  ->  2*ln(L) - k*ln(n)                  (-BIC)
      'aic'          ->  2*ln(L) - 2k                        (-AIC)
      'ebic'         ->  -(BIC + 2*gamma*ln C(p,k))          (-EBIC; Chen & Chen 2008)
      'rebic'        ->  -REBIC  (regression); -EBIC (classif.)
      'llf'          ->  ln(L)   (raw log-likelihood)

    For regression, ln(L) is the Gaussian MLE log-likelihood (sigma^2 profiled out):
        ln L = -n/2 * ln(2*pi*RSS/n) - n/2

    For classification, the sum form of log-loss is used:
        ln L = -sum_i log p(y_i)

    k counts only non-zero coefficients (sparse-aware) plus intercept if
    present.  p_total is the total candidate parameter count.  Both include
    +1 for sigma^2 in regression.

    :param model_factory:     estimator class, instance, or zero-arg callable
    :param X:                 (n, p) full training array
    :param y:                 (n,) target vector
    :param feature_names:     all feature names aligned with X
    :param selected_features: subset to evaluate
    :param task:              'regression' or 'classification'
    :param criterion:         one of 'bic','aic','sic','ebic','rebic','llf'
    :param gamma:             EBIC/REBIC weight in [0,1] (default 1; 0 = BIC)
    :return: (score, log_lik, k_effective, n_samples)  — higher score = better
    """
    from sklearn.metrics import log_loss
    from scipy.special import gammaln

    model_factory = _coerce_factory(model_factory)
    n     = len(y)
    X_sel = _select_columns(X, feature_names, selected_features)
    model = model_factory()
    model.fit(X_sel, y)

    coef = getattr(model, 'coef_', None)
    if coef is not None:
        k_nz    = int(np.count_nonzero(coef))
        p_total = int(np.prod(np.array(coef).shape))
    else:
        k_nz    = len(selected_features)
        p_total = len(selected_features)

    has_intercept = bool(
        getattr(model, 'fit_intercept', False) or
        (getattr(model, 'intercept_', None) is not None and
         np.any(getattr(model, 'intercept_', 0) != 0))
    )
    if has_intercept:
        k_nz    += 1
        p_total += 1

    k = k_nz   # working count; +1 for sigma^2 added below

    if task == 'regression':
        y_pred  = model.predict(X_sel)
        rss     = max(float(np.sum((y - y_pred) ** 2)), 1e-10)
        log_lik = -n / 2.0 * np.log(2.0 * np.pi * rss / n) - n / 2.0
        k       += 1       # sigma^2
        p_total += 1
    else:
        proba   = model.predict_proba(X_sel)
        log_lik = float(-log_loss(y, proba, normalize=False))

    criterion = criterion.lower()

    if criterion in ('bic', 'sic'):
        return 2.0 * log_lik - k * np.log(n), log_lik, k, n

    if criterion == 'aic':
        return 2.0 * log_lik - 2.0 * k, log_lik, k, n

    if criterion == 'llf':
        return log_lik, log_lik, k, n

    # log C(p_total, k) via gammaln — avoids integer overflow for large p_total
    log_comb = float(
        gammaln(p_total + 1) - gammaln(k + 1) - gammaln(p_total - k + 1)
    )
    log_comb = max(log_comb, 0.0)  # guard against floating-point noise near 0

    if criterion == 'ebic':
        bic_pen = k * np.log(n) - 2.0 * log_lik
        return -(bic_pen + 2.0 * gamma * log_comb), log_lik, k, n

    if criterion == 'rebic':
        if task == 'regression':
            rss0  = max(float(np.sum((y - np.mean(y)) ** 2)), 1e-10) / n
            rebic = (n * np.log(rss / n)
                     + k * np.log(n / (2.0 * np.pi))
                     + (k + 2) * np.log(rss0 / rss)
                     + 2.0 * gamma * log_comb)
            return -rebic, log_lik, k, n
        else:
            bic_pen = k * np.log(n) - 2.0 * log_lik
            return -(bic_pen + 2.0 * gamma * log_comb), log_lik, k, n

    raise ValueError(
        f"Unknown criterion '{criterion}'. "
        "Choose from: 'bic', 'aic', 'sic', 'ebic', 'rebic', 'llf'."
    )


#: Public alias — score a feature subset directly.
compute_criterion = _compute_criterion


# ---------------------------------------------------------------------------
# Keep Absolute metric  (paper Section IV, Fig. 3)
# ---------------------------------------------------------------------------

def keep_absolute(model_factory, X, y, feature_names, ordered_feature_names,
                  task='classification', steps=None, cv=None, scoring=None,
                  step_by='fraction'):
    """
    Evaluate a SHAP feature ranking with the Keep Absolute metric.

    For each fraction in steps the model is retrained on the top-d features
    and scored.  The AUC over the resulting curve summarises ranking quality.

    Two scoring modes:

    Cross-validated scorers
        sklearn strings ('r2', 'f1_weighted', ...), callables, or 'llf'.
        Scored via cross_val_score.  cv controls fold count (default 3).
        std is meaningful; the 1-std rule applies normally.

    In-sample information criteria
        'bic', 'aic', 'sic', 'ebic', 'rebic'
        Model fitted ONCE on full training subset; IC evaluated on same n.
        cv is IGNORED — not used, silently.  std = 0 at every step.
        Pass scoring=('ebic', 0.5) to set the EBIC gamma weight.

    :param model_factory:         estimator class, instance, or zero-arg callable.
                                  Accepted forms:
                                    LinearRegression          (class)
                                    Ridge(alpha=0.1)          (instance — cloned each step)
                                    lambda: Ridge(alpha=0.1)  (callable)
    :param X:                     (n, p) array
    :param y:                     (n,) target
    :param feature_names:         list of all feature names aligned with X
    :param ordered_feature_names: SHAP-ranked names (output of shap_select)
    :param task:                  'classification' or 'regression'
    :param steps:                 list of fractions in (0,1]; default depends
                                  on step_by
    :param cv:                    CV folds for cross-validated scorers (default
                                  3).  **Forced to None for in-sample criteria**
                                  (BIC/AIC/EBIC/REBIC): cross_val_score is never
                                  called for those, regardless of what is passed.
    :param scoring:               scorer string, callable, criterion name, or
                                  (criterion, gamma) tuple.
                                  Default: 'f1_weighted' / 'r2'
    :param step_by:               'fraction' (default) 10%..100%, or
                                  'feature' 1..n (ignored when steps given)
    :return: dict with keys
             'fractions' -- list of fractions evaluated
             'scores'    -- mean score at each fraction (higher = better)
             'std'       -- std of scores (0 for in-sample criteria)
             'auc'       -- trapezoidal AUC over fractions vs scores
    """
    model_factory         = _coerce_factory(model_factory)
    feature_names         = list(feature_names)
    ordered_feature_names = list(ordered_feature_names)
    n_features            = len(feature_names)

    unknown = set(ordered_feature_names) - set(feature_names)
    if unknown:
        raise ValueError(
            f"ordered_feature_names contains names not in feature_names: {unknown}. "
            "Ensure shap_select was called on the same dataset."
        )

    if steps is None:
        steps = (
            [i / n_features for i in range(1, n_features + 1)]
            if step_by == 'feature'
            else [round(i / 10, 1) for i in range(1, 11)]
        )
    if scoring is None:
        scoring = 'f1_weighted' if task == 'classification' else 'r2'

    cv_scoring, insample_criterion, gamma = _parse_scoring(scoring, task)

    # Resolve effective CV fold count.
    # In-sample criteria (BIC/AIC/EBIC/REBIC) bypass cross_val_score entirely,
    # so effective_cv is forced to None regardless of what the caller passed.
    # For CV-path scorers, effective_cv uses the caller's value or defaults to 3.
    if insample_criterion is not None:
        effective_cv = None          # not used; made explicit so no ambiguity
    else:
        effective_cv = cv if cv is not None else 3

    X_arr       = np.asarray(X)
    mean_scores = []
    std_scores  = []

    for frac in steps:
        n_keep     = max(1, int(np.ceil(frac * n_features)))
        keep_names = ordered_feature_names[:n_keep]

        if insample_criterion is not None:
            # In-sample path.
            # effective_cv == None — cross_val_score is never called.
            # n = full training set size, as required by IC theory:
            # the penalty term k·ln(n) and ln L̂ must share the same n.
            score, _, _, _ = _compute_criterion(
                model_factory, X_arr, y, feature_names, keep_names,
                task, insample_criterion, gamma,
            )
            mean_scores.append(score)
            std_scores.append(0.0)

        else:
            # CV path — sklearn scorer or llf.
            # effective_cv is the caller's cv value, or 3 by default.
            X_sub     = _select_columns(X_arr, feature_names, keep_names)
            cv_scores = cross_val_score(
                model_factory(), X_sub, y, cv=effective_cv, scoring=cv_scoring
            )
            if np.any(np.isnan(cv_scores)):
                raise ValueError(
                    f"CV scoring produced NaN at fraction {frac} "
                    f"({n_keep} features: {keep_names}). "
                    "Check for rank-deficient subsets or a fit_intercept mismatch."
                )
            mean_scores.append(float(cv_scores.mean()))
            std_scores.append(float(cv_scores.std()))

    mean_scores = np.array(mean_scores)
    std_scores  = np.array(std_scores)

    return {
        'fractions': steps,
        'scores':    mean_scores,
        'std':       std_scores,
        'auc':       float(_trapz(mean_scores, x=steps)),
    }


def select_by_keep_absolute(model_factory, X, y, feature_names,
                             ordered_feature_names, task='classification',
                             steps=None, cv=None, scoring=None,
                             step_by='fraction'):
    """
    Select the smallest feature subset within one standard deviation of the
    peak score (the *1-std rule*).

    Builds the full Keep Absolute sweep then applies:
        threshold = peak_score - std_at_peak
        selected  = smallest fraction whose mean score >= threshold

    For in-sample criteria std = 0 everywhere, so the rule reduces to argmax.

    scoring accepts:
      sklearn strings ('r2', 'f1_weighted', ...)    -> cross-validated
      'llf'                                          -> CV log-likelihood
      'bic', 'aic', 'sic', 'ebic', 'rebic'          -> in-sample IC (cv forced None)
      ('ebic', 0.5)                                  -> in-sample EBIC, gamma=0.5

    :param model_factory:         estimator class, instance, or zero-arg callable.
                                  Accepted forms:
                                    LinearRegression          (class)
                                    Ridge(alpha=0.1)          (instance — cloned each step)
                                    lambda: Ridge(alpha=0.1)  (callable)
    :param X:                     (n, p) array
    :param y:                     (n,) target
    :param feature_names:         list of all feature names
    :param ordered_feature_names: SHAP-ranked names (from shap_select)
    :param task:                  'classification' or 'regression'
    :param steps:                 fractions to evaluate; default 0.1..1.0
    :param cv:                    CV folds for cross-validated scorers (default
                                  3); forced to None (unused) for in-sample IC
    :param scoring:               scorer, criterion name, or (criterion, gamma)
    :param step_by:               'fraction' (default) or 'feature'
    :return: all keep_absolute keys plus:
             'best_fraction'     -- fraction at peak score
             'selected_fraction' -- fraction chosen by 1-std rule (<=best)
             'selected_features' -- list of selected feature names
    """
    ka = keep_absolute(
        model_factory, X, y, feature_names, ordered_feature_names,
        task=task, steps=steps, cv=cv, scoring=scoring, step_by=step_by,
    )

    fractions             = ka['fractions']
    scores                = ka['scores']
    stds                  = ka['std']
    ordered_feature_names = list(ordered_feature_names)
    n_features            = len(ordered_feature_names)

    best_idx  = int(np.argmax(scores))
    threshold = scores[best_idx] - stds[best_idx]
    # Walk forward from index 0; pick the first fraction at or above threshold.
    # argmax on the boolean mask gives the lowest index where condition is True.
    meets     = scores >= threshold
    sel_idx   = int(np.argmax(meets))   # first True; guaranteed >= 0

    selected_frac     = fractions[sel_idx]
    n_keep            = max(1, int(np.ceil(selected_frac * n_features)))
    selected_features = ordered_feature_names[:n_keep]

    ka['best_fraction']     = fractions[best_idx]
    ka['selected_fraction'] = selected_frac
    ka['selected_features'] = selected_features
    return ka


# ---------------------------------------------------------------------------
# Knee detection — Kneeliverse integration
# ---------------------------------------------------------------------------

_KNEE_REGISTRY = {
    'kneedle':   ('kneeliverse.kneedle',   'knee'),
    'dfdt':      ('kneeliverse.dfdt',      'knee'),
    'curvature': ('kneeliverse.curvature', 'knee'),
    'menger':    ('kneeliverse.menger',    'knee'),
    'lmethod':   ('kneeliverse.lmethod',   'knee'),
}

#: Valid knee detection method names.
KNEE_METHODS = list(_KNEE_REGISTRY.keys())


def _deduplicate_curve(fractions, scores):
    """
    Collapse plateaus: keep only the first (minimum-fraction) point of each
    run of equal scores.

    Knee algorithms require strictly varying input.  Polynomial or correlated
    features often produce flat runs where adjacent subsets score identically.
    Collapsing each plateau to its onset gives the algorithm the sharpest signal.
    """
    x = np.asarray(fractions, dtype=float)
    y = np.asarray(scores,    dtype=float)
    score_range = y.max() - y.min()
    atol  = score_range * 1e-6 if score_range > 0 else 1e-12
    keep  = [0]
    for i in range(1, len(y)):
        if not np.isclose(y[i], y[keep[-1]], atol=atol, rtol=0):
            keep.append(i)
    return x[keep], y[keep]


def _infer_curve_shape(x, y):
    """
    Classify the curve as concave or convex by comparing its AUC to the chord.

    For an increasing curve:
      concave (diminishing returns) -> bows above chord -> AUC > chord area
      convex  (accelerating gains)  -> bows below chord -> AUC < chord area

    Knee algorithms are designed for concave curves; convex ones need different
    handling.

    :return: (curve_shape, direction)
             curve_shape in {'concave', 'convex'}
             direction   in {'increasing', 'decreasing'}
    """
    direction  = 'increasing' if y[-1] >= y[0] else 'decreasing'
    chord_area = (y[0] + y[-1]) / 2.0 * (x[-1] - x[0])
    curve_area = float(_trapz(y, x))
    if direction == 'increasing':
        curve = 'concave' if curve_area >= chord_area else 'convex'
    else:
        curve = 'concave' if curve_area <= chord_area else 'convex'
    return curve, direction


def _run_knee_method(fractions, scores, method, **method_kwargs):
    """
    Dispatch a Kneeliverse knee algorithm on the (possibly in-sample IC) curve.

    Preprocessing:
      1. Deduplication — collapses equal-score plateaus.
      2. Shape inference — detects concave vs convex automatically.

    For convex curves (any direction) the algorithm is bypassed: the library
    returns x[-1] of the deduplicated array (start of the final plateau).

    :return: float knee fraction or None if undetectable
    """
    import importlib

    module_path, fn_name = _KNEE_REGISTRY[method]
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        raise ImportError(
            f"Knee method '{method}' requires kneeliverse: pip install kneeliverse"
        )

    x, y = _deduplicate_curve(fractions, scores)
    if len(x) < 3:
        return None

    curve, direction = _infer_curve_shape(x, y)

    if curve == 'convex':
        return float(x[-1])

    if method == 'kneedle':
        from kneeliverse.kneedle import PeakDetection
        p       = method_kwargs.get('p', PeakDetection.All)
        points  = np.column_stack([x, y])
        indices = mod.knees(points, p=p)
        if len(indices) == 0:
            return None
        return float(x[indices[0]])

    fn = getattr(mod, fn_name)
    import inspect
    sig         = inspect.signature(fn)
    call_kwargs = {}
    if 'curve'     in sig.parameters: call_kwargs['curve']     = method_kwargs.get('curve',     curve)
    if 'direction' in sig.parameters: call_kwargs['direction'] = method_kwargs.get('direction', direction)
    for k, v in method_kwargs.items():
        if k not in ('curve', 'direction', 'p'):
            call_kwargs[k] = v

    points = np.column_stack([x, y])
    idx    = fn(points, **call_kwargs) if call_kwargs else fn(points)

    if idx is None or (isinstance(idx, (int, np.integer)) and idx < 0):
        return None
    return float(x[int(idx)])


def select_by_knee_detection(model_factory, X, y, feature_names,
                              ordered_feature_names, task='classification',
                              steps=None, cv=None, scoring=None,
                              method='kneedle', step_by='feature',
                              log_transform_scores=False,
                              **method_kwargs):
    """
    Find the point of diminishing returns on the score-vs-fraction curve using
    a knee-detection algorithm from Kneeliverse.

    The same sweep as keep_absolute is run.  Instead of the 1-std rule, a knee
    algorithm identifies the elbow of the performance curve.

    Curve shape is auto-detected:
      concave (typical)        -> knee algorithm runs normally
      convex  (polynomial etc) -> algorithm bypassed; start of final plateau
                                  returned directly

    scoring accepts the same values as keep_absolute:
      sklearn strings / callables, 'llf' (CV), or in-sample IC names / tuples.
      For in-sample criteria cv is ignored and the knee is detected on the IC curve.

    When using in-sample criteria, consider setting log_transform_scores=True
    to compress unbounded IC values into a log-scale range before passing them
    to the knee algorithm.  See the log_transform_scores parameter below.

    Requires: pip install kneeliverse

    Available methods
    -----------------
    'kneedle'   (default) Kneedle (Satopaa et al. 2011)
    'dfdt'                Dynamic First Derivative Thresholding
    'curvature'           Discrete curvature
    'menger'              Menger curvature
    'lmethod'             L-method (two-segment linear fit)

    :param model_factory:         estimator class, instance, or zero-arg callable.
                                  Accepted forms:
                                    LinearRegression          (class)
                                    Ridge(alpha=0.1)          (instance — cloned each step)
                                    lambda: Ridge(alpha=0.1)  (callable)
    :param X:                     (n, p) array
    :param y:                     (n,) target
    :param feature_names:         list of all feature names
    :param ordered_feature_names: SHAP-ranked names (from shap_select)
    :param task:                  'classification' or 'regression'
    :param steps:                 fractions to evaluate (default: one per feature)
    :param cv:                    CV folds for cross-validated scorers (default
                                  3); forced to None (unused) for in-sample IC
    :param scoring:               scorer, criterion name, or (criterion, gamma)
    :param method:                knee algorithm (default 'kneedle')
    :param step_by:               'feature' (default) or 'fraction'
    :param log_transform_scores:  if True, apply log1p(scores − scores.min())
                                  to the sweep scores before passing them to the
                                  knee algorithm.  The raw scores are stored in
                                  the result dict unchanged; only the curve seen
                                  by the knee algorithm is transformed.
                                  Useful when scoring='bic'/'aic'/etc. to
                                  compress large unbounded IC values into a
                                  bounded log-scale range that knee algorithms
                                  handle better.  Default False.
    :param method_kwargs:         extra kwargs forwarded to Kneeliverse
    :return: all keep_absolute keys plus:
             'knee_fraction'       -- fraction at detected knee
             'selected_fraction'   -- same as knee_fraction (or fallback to peak)
             'selected_features'   -- list of selected feature names
             'method'              -- algorithm used
             'curve_shape'         -- 'concave' or 'convex'
             'curve_direction'     -- 'increasing' or 'decreasing'
             'log_transform_scores'-- bool: whether log transform was applied
    """
    if method not in _KNEE_REGISTRY:
        raise ValueError(f"Unknown knee method '{method}'. Choose from: {KNEE_METHODS}")

    ka = keep_absolute(
        model_factory, X, y, feature_names, ordered_feature_names,
        task=task, steps=steps, cv=cv, scoring=scoring, step_by=step_by,
    )

    fractions             = np.array(ka['fractions'])
    scores                = np.array(ka['scores'])
    ordered_feature_names = list(ordered_feature_names)
    n_features            = len(ordered_feature_names)

    # Optionally transform scores before passing to the knee algorithm.
    # The raw scores (stored in ka['scores']) are preserved; only the curve
    # presented to _run_knee_method is transformed.
    knee_scores = _log_transform_scores(scores) if log_transform_scores else scores

    knee_frac = _run_knee_method(fractions, knee_scores, method, **method_kwargs)

    if knee_frac is None:
        warnings.warn(
            f"Knee method '{method}' could not find a knee (curve may be flat). "
            "Falling back to peak-score fraction.  "
            "Try a different method or use select_by_keep_absolute instead.",
            UserWarning, stacklevel=2,
        )
        knee_frac = float(fractions[int(np.argmax(scores))])

    x_dd, y_dd = _deduplicate_curve(fractions, scores)
    curve_shape, curve_direction = (
        _infer_curve_shape(x_dd, y_dd) if len(x_dd) >= 3 else ('unknown', 'unknown')
    )

    n_keep            = max(1, int(np.ceil(knee_frac * n_features)))
    selected_features = ordered_feature_names[:n_keep]

    ka['knee_fraction']        = knee_frac
    ka['selected_fraction']    = knee_frac
    ka['selected_features']    = selected_features
    ka['method']               = method
    ka['curve_shape']          = curve_shape
    ka['curve_direction']      = curve_direction
    ka['log_transform_scores'] = bool(log_transform_scores)
    return ka


# ---------------------------------------------------------------------------
# Auto-select
# ---------------------------------------------------------------------------

def auto_select(
    model_factory, X, y, feature_names, ordered_feature_names,
    task='classification',
    steps=None,
    cv=None,
    scoring=None,
    criterion=None,
    knee_method='kneedle',
    step_by_absolute='fraction',
    step_by_knee='feature',
    log_transform_scores=False,
    **knee_kwargs,
):
    """
    Run select_by_keep_absolute and select_by_knee_detection on the same
    SHAP-ordered features and return the better result.

    Two operating modes, controlled by the ``criterion`` argument:

    **Unified mode** (``criterion=None``, default)
        Both strategies share the same ``scoring`` and ``cv`` arguments.
        Scores are on the same scale so direct comparison is valid.
        The winner is whichever sub-result has the higher score at its
        selected fraction.  This is the behaviour of earlier versions.

    **Split mode** (``criterion`` is set)
        ``select_by_keep_absolute`` uses ``scoring`` + ``cv`` (CV path),
        giving a meaningful 1-std rule with real fold variance.

        ``select_by_knee_detection`` uses ``criterion`` (in-sample IC path),
        giving a principled elbow in IC space; ``cv`` is forced to None
        for this leg.

        Because the two legs now score on different scales (e.g. r2 vs −BIC),
        direct comparison of their sweep scores is invalid.  Instead the winner
        is decided by evaluating ``criterion`` **in-sample** on *both* selected
        subsets and picking the better IC value.  The result dict exposes
        ``absolute_criterion_score`` and ``knee_criterion_score`` for this
        comparison, in addition to the per-curve scores.

    ``scoring`` accepts any sklearn scorer string (``'r2'``, ``'f1_weighted'``,
    …), ``'llf'`` (CV log-likelihood), an in-sample criterion name, or a
    ``(criterion, gamma)`` tuple — same as ``keep_absolute``.

    ``criterion`` accepts an in-sample criterion name (``'bic'``, ``'aic'``,
    ``'sic'``, ``'ebic'``, ``'rebic'``) or a ``(criterion, gamma)`` tuple.
    When set, it is used exclusively for knee detection and winner comparison.

    :param model_factory:     estimator class, instance, or zero-arg callable.
                              Accepted forms:
                                LinearRegression          (class)
                                Ridge(alpha=0.1)          (instance — cloned each step)
                                lambda: Ridge(alpha=0.1)  (callable)
    :param X:                 (n, p) array
    :param y:                 (n,) target
    :param feature_names:     list of all feature names
    :param ordered_feature_names: SHAP-ranked names (from shap_select)
    :param task:              'classification' or 'regression'
    :param steps:             fraction steps (default: auto)
    :param cv:                CV folds for keep_absolute (default 3); always
                              forwarded to select_by_keep_absolute.  Forced to
                              None inside select_by_knee_detection when
                              ``criterion`` is set.
    :param scoring:           scorer for keep_absolute (and for knee when
                              criterion=None).  sklearn string, 'llf', IC name,
                              or (criterion, gamma) tuple.
                              Default: 'f1_weighted' / 'r2'
    :param criterion:         in-sample IC for knee detection and winner
                              comparison (split mode).  IC name or tuple.
                              Default None → unified mode.
    :param knee_method:       Kneeliverse method (default 'kneedle')
    :param step_by_absolute:  step granularity for keep_absolute
                              ('fraction' default)
    :param step_by_knee:      step granularity for knee detection
                              ('feature' default)
    :param log_transform_scores: if True, apply log1p(scores − min) to the
                              sweep scores before running the knee algorithm.
                              Forwarded to select_by_knee_detection only.
                              Has no effect on keep_absolute or winner
                              comparison.  Useful with in-sample IC scoring
                              to compress large unbounded values.
                              Default False.
    :param knee_kwargs:       extra kwargs forwarded to select_by_knee_detection
    :return: dict with keys:

        Always present
        ~~~~~~~~~~~~~~
        ``selected_features``        winning feature list
        ``winner``                   'absolute' or 'knee'
        ``scoring``                  scoring arg used for keep_absolute
        ``criterion``                criterion arg (None in unified mode)
        ``absolute_features``        features from keep_absolute
        ``knee_features``            features from knee detection
        ``absolute_score``           score at selected fraction on the
                                     keep_absolute sweep curve
        ``knee_score``               score at selected fraction on the
                                     knee sweep curve
        ``absolute_result``          full dict from select_by_keep_absolute
        ``knee_result``              full dict from select_by_knee_detection

        Split mode only (criterion is not None)
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ``absolute_criterion_score`` IC score of absolute-selected subset
        ``knee_criterion_score``     IC score of knee-selected subset
        ``criterion_gamma``          gamma used for IC comparison
    """
    # --- parse criterion arg (split mode) -----------------------------------
    knee_scoring   = scoring   # what knee detection uses
    crit_name      = None      # IC name for winner comparison
    crit_gamma     = 1.0

    if criterion is not None:
        if isinstance(criterion, tuple):
            if (len(criterion) != 2 or
                    not isinstance(criterion[0], str) or
                    criterion[0].lower() not in INSAMPLE_CRITERIA):
                raise ValueError(
                    f"criterion tuple must be (name, gamma) where name in "
                    f"{sorted(INSAMPLE_CRITERIA)}, got {criterion!r}"
                )
            crit_name, crit_gamma = criterion[0].lower(), float(criterion[1])
        elif isinstance(criterion, str):
            if criterion.lower() not in INSAMPLE_CRITERIA:
                raise ValueError(
                    f"criterion must be one of {sorted(INSAMPLE_CRITERIA)} "
                    f"or a (name, gamma) tuple, got {criterion!r}"
                )
            crit_name = criterion.lower()
        else:
            raise ValueError(
                f"criterion must be a string or (name, gamma) tuple, "
                f"got {type(criterion)}"
            )
        knee_scoring = crit_name   # knee uses in-sample IC, cv forced None

    # --- run both strategies ------------------------------------------------
    abs_result = select_by_keep_absolute(
        model_factory, X, y, feature_names, ordered_feature_names,
        task=task, steps=steps, cv=cv, scoring=scoring,
        step_by=step_by_absolute,
    )
    knee_result = select_by_knee_detection(
        model_factory, X, y, feature_names, ordered_feature_names,
        task=task, steps=steps, cv=cv, scoring=knee_scoring,
        method=knee_method, step_by=step_by_knee,
        log_transform_scores=log_transform_scores,
        **knee_kwargs,
    )

    def _score_at(result):
        fracs = np.array(result['fractions'])
        idx   = int(np.argmin(np.abs(fracs - result['selected_fraction'])))
        return float(result['scores'][idx])

    abs_score  = _score_at(abs_result)
    knee_score = _score_at(knee_result)

    # --- winner comparison --------------------------------------------------
    if crit_name is not None:
        # Split mode: scores are on different scales.  Evaluate criterion
        # in-sample on both selected subsets and compare those IC values.
        X_arr = np.asarray(X)
        abs_crit_score, _, _, _ = _compute_criterion(
            model_factory, X_arr, y, feature_names,
            abs_result['selected_features'],
            task, crit_name, crit_gamma,
        )
        knee_crit_score, _, _, _ = _compute_criterion(
            model_factory, X_arr, y, feature_names,
            knee_result['selected_features'],
            task, crit_name, crit_gamma,
        )
        winner = 'absolute' if abs_crit_score >= knee_crit_score else 'knee'
    else:
        # Unified mode: same scorer → direct comparison valid.
        abs_crit_score  = None
        knee_crit_score = None
        winner = 'absolute' if abs_score >= knee_score else 'knee'

    out = {
        'selected_features': list((abs_result if winner == 'absolute' else knee_result)['selected_features']),
        'winner':            winner,
        'scoring':           scoring,
        'criterion':         criterion,
        'absolute_features': list(abs_result['selected_features']),
        'knee_features':     list(knee_result['selected_features']),
        'absolute_score':    abs_score,
        'knee_score':        knee_score,
        'absolute_result':   abs_result,
        'knee_result':       knee_result,
    }
    if crit_name is not None:
        out['absolute_criterion_score'] = abs_crit_score
        out['knee_criterion_score']     = knee_crit_score
        out['criterion_gamma']          = crit_gamma
    return out


# ---------------------------------------------------------------------------
# Threshold / top-k selection
# ---------------------------------------------------------------------------

def shap_threshold_select(model, X_train, feature_names,
                           task='classification', background_size=0.1,
                           threshold=None, top_k=None, random_state=None):
    """
    Select features by importance threshold or top-k count.

    Provide at most one of threshold or top_k.

    :param random_state: forwarded to shap_select — controls background
                         sampling for KernelExplainer models.
                         None / int / numpy.random.Generator.
    :return: (selected_names, selected_importance,
              all_ordered_names, all_ordered_importance)
    """
    if threshold is not None and top_k is not None:
        raise ValueError("Provide at most one of `threshold` or `top_k`.")
    ordered_names, ordered_importance = shap_select(
        model, X_train, feature_names, task=task,
        background_size=background_size, random_state=random_state,
    )
    if threshold is not None:
        mask = ordered_importance >= threshold
    elif top_k is not None:
        top_k = min(top_k, len(ordered_names))
        mask  = np.zeros(len(ordered_names), dtype=bool)
        mask[:top_k] = True
    else:
        mask = np.ones(len(ordered_names), dtype=bool)
    return ordered_names[mask], ordered_importance[mask], ordered_names, ordered_importance


# ---------------------------------------------------------------------------
# Dataset filtering
# ---------------------------------------------------------------------------

def apply_feature_selection(X, all_feature_names, selected_feature_names):
    """
    Filter a dataset to retain only the selected feature columns.

    Supports numpy arrays and pandas DataFrames.

    :param X:                      (n, p) array or DataFrame
    :param all_feature_names:      ordered list of all feature names in X
    :param selected_feature_names: list of feature names to keep
    :return: filtered array or DataFrame
    """
    all_feature_names      = list(all_feature_names)
    selected_feature_names = list(selected_feature_names)
    try:
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X[selected_feature_names]
    except ImportError:
        pass
    try:
        name_to_idx = {n: i for i, n in enumerate(all_feature_names)}
        indices = [name_to_idx[name] for name in selected_feature_names]
    except KeyError as e:
        raise ValueError(
            f"Feature not found: {e}.  "
            "Check selected_feature_names is a subset of all_feature_names."
        ) from e
    return np.asarray(X)[:, indices]
