"""
Tests for shap_selection v1.0.0.

Synthetic data is used throughout so the suite runs quickly without large
datasets.  Each test class covers one module-level concern.

Test classes
------------
TestModelFactory            _coerce_factory — class, instance, callable, bad input
TestRandomState             random_state / _resolve_rng — reproducibility
TestShapSelect              shap_select ranking invariants
TestKeepAbsolute            keep_absolute sweep outputs
TestKeepAbsoluteCvPath      CV path (r2, llf) — std is meaningful
TestKeepAbsoluteInsample    In-sample IC path — cv ignored, std = 0
TestSelectByKeepAbsolute    1-std rule selection
TestLogTransform            _log_transform_scores helper
TestSelectByKneeDetection   knee detection (skipped without kneeliverse)
TestComputeCriterion        compute_criterion mathematical correctness
TestAutoSelect              auto_select end-to-end (unified + split modes)
TestShapThresholdSelect     threshold / top-k selection
TestApplyFeatureSelection   dataset filtering utility
TestScoringDispatch         _parse_scoring routing correctness
TestDefaults                default parameter behaviour
TestEdgeCases               single-feature, numpy array names, etc.
"""

import warnings
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression, make_classification

from shap_selection import (
    shap_select,
    keep_absolute,
    select_by_keep_absolute,
    select_by_knee_detection,
    auto_select,
    compute_criterion,
    shap_threshold_select,
    apply_feature_selection,
    KNEE_METHODS,
    INSAMPLE_CRITERIA,
    CRITERION_SCORERS,
)
import shap_selection._core as _core


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reg_data():
    X, y = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=42)
    feat  = np.array([f"f{i}" for i in range(X.shape[1])])
    split = 160
    return X[:split], X[split:], y[:split], y[split:], feat


@pytest.fixture(scope="module")
def cls_data():
    X, y = make_classification(
        n_samples=200, n_features=8, n_informative=4, random_state=42
    )
    feat  = np.array([f"f{i}" for i in range(X.shape[1])])
    split = 160
    return X[:split], X[split:], y[:split], y[split:], feat


@pytest.fixture(scope="module")
def fitted_linear(reg_data):
    X_tr, _, y_tr, _, _ = reg_data
    m = LinearRegression()
    m.fit(X_tr, y_tr)
    return m


@pytest.fixture(scope="module")
def fitted_tree(reg_data):
    X_tr, _, y_tr, _, _ = reg_data
    m = DecisionTreeRegressor(max_depth=3, random_state=42)
    m.fit(X_tr, y_tr)
    return m


@pytest.fixture(scope="module")
def ordered_reg(fitted_linear, reg_data):
    X_tr, _, _, _, feat = reg_data
    names, _ = shap_select(fitted_linear, X_tr, feat, task='regression')
    return names


# ---------------------------------------------------------------------------
# TestModelFactory  — _coerce_factory accepts class / instance / callable
# ---------------------------------------------------------------------------

class TestModelFactory:
    """
    _coerce_factory must normalise all three accepted forms to a zero-arg
    callable and reject everything else.
    """

    def _make_data(self):
        rng  = np.random.default_rng(0)
        X    = rng.standard_normal((60, 4))
        y    = X[:, 0] * 2 + rng.standard_normal(60) * 0.1
        feat = [f"f{i}" for i in range(4)]
        m    = LinearRegression().fit(X, y)
        names, _ = shap_select(m, X, feat, task='regression')
        return X, y, feat, names

    # --- _coerce_factory unit tests ----------------------------------------

    def test_class_passthrough(self):
        """An estimator class is returned unchanged."""
        factory = _core._coerce_factory(LinearRegression)
        assert factory is LinearRegression
        assert isinstance(factory(), LinearRegression)

    def test_instance_becomes_callable(self):
        """An estimator instance is wrapped in a clone-callable."""
        instance = Ridge(alpha=2.0)
        factory  = _core._coerce_factory(instance)
        assert callable(factory)
        m1 = factory()
        m2 = factory()
        assert m1 is not m2, "factory must return independent objects"
        assert m1.alpha == 2.0

    def test_instance_not_mutated(self):
        """Fitting the cloned model must not affect the original instance."""
        rng      = np.random.default_rng(1)
        X        = rng.standard_normal((30, 2))
        y        = X[:, 0] + 0.1
        instance = Ridge(alpha=3.0)
        factory  = _core._coerce_factory(instance)
        m        = factory()
        m.fit(X, y)
        assert not hasattr(instance, 'coef_'), \
            "fitting the clone must not fit the original instance"

    def test_fitted_instance_produces_unfitted_clone(self):
        """Even a pre-fitted instance must produce fresh unfitted clones."""
        rng      = np.random.default_rng(2)
        X        = rng.standard_normal((30, 2))
        y        = X[:, 0] + 0.1
        fitted   = LinearRegression().fit(X, y)
        factory  = _core._coerce_factory(fitted)
        fresh    = factory()
        assert not hasattr(fresh, 'coef_'), \
            "clone of a fitted instance must be unfitted"

    def test_callable_passthrough(self):
        """A zero-arg callable (lambda) is returned unchanged."""
        fn      = lambda: Ridge(alpha=5.0)
        factory = _core._coerce_factory(fn)
        assert factory is fn

    def test_non_callable_raises(self):
        with pytest.raises(TypeError, match="model_factory"):
            _core._coerce_factory(42)

    def test_non_callable_string_raises(self):
        with pytest.raises(TypeError):
            _core._coerce_factory("LinearRegression")

    # --- integration: all three forms work end-to-end ----------------------

    def test_class_form_keep_absolute(self, reg_data, ordered_reg):
        """Passing LinearRegression (class) must work."""
        X_tr, _, y_tr, _, feat = reg_data
        result = keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=[0.5, 1.0], cv=2)
        assert 'scores' in result

    def test_instance_form_keep_absolute(self, reg_data, ordered_reg):
        """Passing Ridge(alpha=0.5) (instance) must work identically to the lambda form."""
        X_tr, _, y_tr, _, feat = reg_data
        result = keep_absolute(Ridge(alpha=0.5), X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=[0.5, 1.0], cv=2)
        assert 'scores' in result
        assert len(result['scores']) == 2

    def test_lambda_form_keep_absolute(self, reg_data, ordered_reg):
        """Passing lambda: Ridge(alpha=0.5) must still work."""
        X_tr, _, y_tr, _, feat = reg_data
        result = keep_absolute(lambda: Ridge(alpha=0.5), X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=[0.5, 1.0], cv=2)
        assert 'scores' in result

    def test_instance_form_select_by_keep_absolute(self):
        X, y, feat, names = self._make_data()
        result = select_by_keep_absolute(Ridge(alpha=1.0), X, y, feat, names,
                                         task='regression', cv=2)
        assert set(result['selected_features']).issubset(set(feat))

    def test_instance_form_compute_criterion(self, reg_data, ordered_reg):
        """compute_criterion must accept an estimator instance."""
        X_tr, _, y_tr, _, feat = reg_data
        sel = list(ordered_reg[:4])
        score, llf, k, n = compute_criterion(
            Ridge(alpha=1.0), X_tr, y_tr, feat, sel,
            task='regression', criterion='bic')
        assert np.isfinite(score)

    def test_class_and_instance_agree(self, reg_data, ordered_reg):
        """LinearRegression (class) and LinearRegression() (instance) must give
        the same sweep scores since clone of a default instance == the class."""
        X_tr, _, y_tr, _, feat = reg_data
        steps  = [0.25, 0.5, 0.75, 1.0]
        r_cls  = keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=steps, cv=3)
        r_inst = keep_absolute(LinearRegression(), X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=steps, cv=3)
        np.testing.assert_allclose(r_cls['scores'], r_inst['scores'], rtol=1e-5,
                                   err_msg="class and default instance should give same scores")


# ---------------------------------------------------------------------------
# TestRandomState  — _resolve_rng + random_state on shap_select /
#                    shap_threshold_select
# ---------------------------------------------------------------------------

class TestRandomState:
    """
    random_state controls background sampling inside _build_explainer for the
    KernelExplainer path (non-linear, non-tree models, len(X) >= 500).

    We test _resolve_rng in isolation, then verify the reproducibility contract
    end-to-end by patching _build_explainer to capture the rng passed to it.
    Linear / tree models are verified to be unaffected.
    """

    # --- _resolve_rng unit tests -------------------------------------------

    def test_none_returns_generator(self):
        rng = _core._resolve_rng(None)
        assert isinstance(rng, np.random.Generator)

    def test_int_returns_generator(self):
        rng = _core._resolve_rng(42)
        assert isinstance(rng, np.random.Generator)

    def test_int_is_reproducible(self):
        """Two Generators with the same seed must produce identical draws."""
        rng1 = _core._resolve_rng(0)
        rng2 = _core._resolve_rng(0)
        np.testing.assert_array_equal(rng1.integers(0, 100, size=10),
                                       rng2.integers(0, 100, size=10))

    def test_generator_passthrough(self):
        gen = np.random.default_rng(7)
        assert _core._resolve_rng(gen) is gen

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="random_state"):
            _core._resolve_rng("42")

    def test_invalid_float_raises(self):
        with pytest.raises(TypeError, match="random_state"):
            _core._resolve_rng(3.14)

    # --- _build_explainer receives rng and uses it -------------------------

    def test_build_explainer_background_reproducible(self):
        """
        Two calls to _build_explainer with the same seed must sample identical
        background rows when len(X) >= 500.
        """
        import shap as _shap
        from unittest.mock import patch, MagicMock

        rng = np.random.default_rng

        # Synthetic large array to trigger sampling branch
        X_big = np.random.default_rng(0).standard_normal((600, 4))
        dummy_model = MagicMock()
        dummy_model.predict = MagicMock(return_value=np.zeros(600))

        captured = {}

        original_KE = _shap.KernelExplainer

        def mock_ke(predict_fn, background):
            captured['background'] = background
            m = MagicMock()
            m.shap_values = MagicMock(return_value=np.zeros((600, 4)))
            return m

        with patch.object(_shap, 'KernelExplainer', mock_ke):
            # Same seed → same background
            _core._build_explainer(dummy_model, X_big, background_size=0.1,
                                   rng=rng(99))
            bg1 = captured['background'].copy()

            _core._build_explainer(dummy_model, X_big, background_size=0.1,
                                   rng=rng(99))
            bg2 = captured['background'].copy()

        np.testing.assert_array_equal(bg1, bg2,
            err_msg="Same seed must produce identical background rows")

    def test_build_explainer_different_seeds_differ(self):
        """Different seeds should (almost certainly) produce different backgrounds."""
        import shap as _shap
        from unittest.mock import patch, MagicMock

        rng = np.random.default_rng
        X_big = np.random.default_rng(1).standard_normal((600, 4))
        dummy_model = MagicMock()
        captured = {}

        def mock_ke(predict_fn, background):
            captured['background'] = background.copy()
            m = MagicMock()
            m.shap_values = MagicMock(return_value=np.zeros((600, 4)))
            return m

        with patch.object(_shap, 'KernelExplainer', mock_ke):
            _core._build_explainer(dummy_model, X_big, rng=rng(1))
            bg1 = captured['background']
            _core._build_explainer(dummy_model, X_big, rng=rng(2))
            bg2 = captured['background']

        # With 600 rows sampling 60, different seeds will almost always differ
        assert not np.array_equal(bg1, bg2), \
            "Different seeds should produce different backgrounds"

    # --- shap_select passes rng correctly ----------------------------------

    def test_shap_select_random_state_int_accepted(self, fitted_linear, reg_data):
        """shap_select(random_state=42) must run without error."""
        X_tr, _, _, _, feat = reg_data
        names, imp = shap_select(fitted_linear, X_tr, feat,
                                  task='regression', random_state=42)
        assert len(names) == len(feat)

    def test_shap_select_random_state_none_accepted(self, fitted_linear, reg_data):
        X_tr, _, _, _, feat = reg_data
        names, imp = shap_select(fitted_linear, X_tr, feat,
                                  task='regression', random_state=None)
        assert len(names) == len(feat)

    def test_shap_select_random_state_generator_accepted(self, fitted_linear, reg_data):
        X_tr, _, _, _, feat = reg_data
        gen = np.random.default_rng(5)
        names, imp = shap_select(fitted_linear, X_tr, feat,
                                  task='regression', random_state=gen)
        assert len(names) == len(feat)

    def test_shap_select_invalid_random_state_raises(self, fitted_linear, reg_data):
        X_tr, _, _, _, feat = reg_data
        with pytest.raises(TypeError, match="random_state"):
            shap_select(fitted_linear, X_tr, feat,
                        task='regression', random_state="bad")

    def test_shap_threshold_select_forwards_random_state(self, fitted_linear, reg_data):
        """shap_threshold_select must accept and forward random_state."""
        X_tr, _, _, _, feat = reg_data
        sel, sel_imp, _, _ = shap_threshold_select(
            fitted_linear, X_tr, feat, task='regression',
            top_k=3, random_state=42)
        assert len(sel) == 3

    def test_linear_model_unaffected_by_random_state(self, fitted_linear, reg_data):
        """LinearRegression uses LinearExplainer — random_state has no effect.
        Results must be identical regardless of seed."""
        X_tr, _, _, _, feat = reg_data
        names0, imp0 = shap_select(fitted_linear, X_tr, feat,
                                    task='regression', random_state=0)
        names1, imp1 = shap_select(fitted_linear, X_tr, feat,
                                    task='regression', random_state=99)
        np.testing.assert_array_equal(names0, names1)
        np.testing.assert_allclose(imp0, imp1)

    def test_tree_model_unaffected_by_random_state(self, fitted_tree, reg_data):
        """DecisionTree uses TreeExplainer — random_state has no effect."""
        X_tr, _, _, _, feat = reg_data
        names0, imp0 = shap_select(fitted_tree, X_tr, feat,
                                    task='regression', random_state=0)
        names1, imp1 = shap_select(fitted_tree, X_tr, feat,
                                    task='regression', random_state=99)
        np.testing.assert_array_equal(names0, names1)
        np.testing.assert_allclose(imp0, imp1)


# ---------------------------------------------------------------------------
# TestShapSelect
# ---------------------------------------------------------------------------

class TestShapSelect:
    def test_returns_all_features(self, fitted_linear, reg_data):
        X_tr, _, _, _, feat = reg_data
        names, imp = shap_select(fitted_linear, X_tr, feat, task='regression')
        assert len(names) == len(feat)
        assert len(imp)   == len(feat)

    def test_sorted_descending(self, fitted_linear, reg_data):
        X_tr, _, _, _, feat = reg_data
        _, imp = shap_select(fitted_linear, X_tr, feat, task='regression')
        assert list(imp) == sorted(imp, reverse=True)

    def test_importance_nonnegative(self, fitted_linear, reg_data):
        X_tr, _, _, _, feat = reg_data
        _, imp = shap_select(fitted_linear, X_tr, feat, task='regression')
        assert np.all(imp >= 0)

    def test_feature_set_unchanged(self, fitted_linear, reg_data):
        X_tr, _, _, _, feat = reg_data
        names, _ = shap_select(fitted_linear, X_tr, feat, task='regression')
        assert set(names) == set(feat)

    def test_works_with_tree(self, fitted_tree, reg_data):
        X_tr, _, _, _, feat = reg_data
        names, imp = shap_select(fitted_tree, X_tr, feat, task='regression')
        assert len(names) == len(feat)
        assert list(imp) == sorted(imp, reverse=True)

    def test_works_with_ridge(self, reg_data):
        X_tr, _, y_tr, _, feat = reg_data
        m = Ridge()
        m.fit(X_tr, y_tr)
        names, _ = shap_select(m, X_tr, feat, task='regression')
        assert len(names) == len(feat)

    def test_accepts_numpy_feature_names(self, fitted_linear, reg_data):
        X_tr, _, _, _, feat = reg_data
        names, _ = shap_select(fitted_linear, X_tr, np.array(feat), task='regression')
        assert len(names) == len(feat)


# ---------------------------------------------------------------------------
# TestKeepAbsolute
# ---------------------------------------------------------------------------

class TestKeepAbsolute:
    def test_returns_expected_keys(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=[0.5, 1.0], cv=2)
        assert {'fractions', 'scores', 'std', 'auc'}.issubset(result.keys())

    def test_length_matches_steps(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        steps  = [0.3, 0.6, 1.0]
        result = keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=steps, cv=2)
        assert len(result['scores']) == 3
        assert len(result['std'])    == 3

    def test_auc_is_scalar(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=[0.5, 1.0], cv=2)
        assert isinstance(result['auc'], float)

    def test_step_by_feature(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                               task='regression', cv=2, step_by='feature')
        assert len(result['scores']) == len(feat)

    def test_unknown_feature_raises(self, reg_data):
        X_tr, _, y_tr, _, feat = reg_data
        bad = np.array(['f0', 'f1', 'UNKNOWN'])
        with pytest.raises(ValueError, match="UNKNOWN"):
            keep_absolute(LinearRegression, X_tr, y_tr, feat, bad,
                          task='regression', steps=[0.5, 1.0], cv=2)


# ---------------------------------------------------------------------------
# TestKeepAbsoluteCvPath  — CV scorers: std is meaningful
# ---------------------------------------------------------------------------

class TestKeepAbsoluteCvPath:
    def test_r2_std_is_finite_nonnegative(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=[0.5, 1.0], cv=3)
        assert np.all(np.isfinite(result['std']))
        assert np.all(result['std'] >= 0)

    def test_llf_is_cv_path(self, reg_data, ordered_reg):
        """llf should flow through cross_val_score, not the in-sample path."""
        X_tr, _, y_tr, _, feat = reg_data
        result = keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=[0.5, 1.0], cv=2,
                               scoring='llf')
        assert not np.any(np.isnan(result['scores']))
        # llf through CV — std can be non-zero
        assert np.all(result['std'] >= 0)

    def test_cv_param_respected(self, reg_data, ordered_reg):
        """cv= is passed through to cross_val_score exactly for CV-path scorers."""
        from unittest.mock import patch
        X_tr, _, y_tr, _, feat = reg_data
        captured = {}
        orig = _core.cross_val_score

        def mock_cvs(est, X, y, cv=None, scoring=None, **kw):
            captured['cv'] = cv
            return orig(est, X, y, cv=cv, scoring=scoring, **kw)

        with patch.object(_core, 'cross_val_score', mock_cvs):
            keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                          task='regression', steps=[1.0], cv=5)
        assert captured['cv'] == 5

    def test_cv_param_ignored_for_insample(self, reg_data, ordered_reg):
        """For in-sample IC, cv is forced to None: cross_val_score must never be called."""
        from unittest.mock import patch
        X_tr, _, y_tr, _, feat = reg_data
        called = []

        def mock_cvs(*args, **kw):
            called.append(True)
            raise AssertionError("cross_val_score must not be called for in-sample IC")

        with patch.object(_core, 'cross_val_score', mock_cvs):
            keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                          task='regression', steps=[0.5, 1.0], cv=99,
                          scoring='bic')
        assert not called, "cross_val_score was called for an in-sample criterion"


# ---------------------------------------------------------------------------
# TestKeepAbsoluteInsample  — in-sample criteria: cv ignored, std = 0
# ---------------------------------------------------------------------------

class TestKeepAbsoluteInsample:
    @pytest.mark.parametrize("crit", sorted(INSAMPLE_CRITERIA))
    def test_std_is_zero(self, reg_data, ordered_reg, crit):
        """In-sample criteria bypass CV entirely — std must be exactly 0."""
        X_tr, _, y_tr, _, feat = reg_data
        result = keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=[0.5, 1.0], cv=99,
                               scoring=crit)
        assert np.allclose(result['std'], 0.0), f"std != 0 for criterion={crit}"

    @pytest.mark.parametrize("crit", sorted(INSAMPLE_CRITERIA))
    def test_scores_finite(self, reg_data, ordered_reg, crit):
        X_tr, _, y_tr, _, feat = reg_data
        result = keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                               scoring=crit)
        assert not np.any(np.isnan(result['scores']))
        assert not np.any(np.isinf(result['scores']))

    def test_gamma_tuple_syntax(self, reg_data, ordered_reg):
        """(criterion, gamma) tuple must work without error."""
        X_tr, _, y_tr, _, feat = reg_data
        result = keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                               task='regression', steps=[0.5, 1.0],
                               scoring=('ebic', 0.5))
        assert np.allclose(result['std'], 0.0)
        assert not np.any(np.isnan(result['scores']))


# ---------------------------------------------------------------------------
# TestSelectByKeepAbsolute
# ---------------------------------------------------------------------------

class TestSelectByKeepAbsolute:
    def test_returns_expected_keys(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = select_by_keep_absolute(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.5, 1.0], cv=2)
        for key in ('selected_features', 'selected_fraction', 'best_fraction',
                    'scores', 'std', 'auc', 'fractions'):
            assert key in result, f"missing key: {key}"

    def test_selected_is_subset(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = select_by_keep_absolute(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.5, 1.0], cv=2)
        assert set(result['selected_features']).issubset(set(feat))
        assert len(result['selected_features']) >= 1

    def test_selected_lte_best(self, reg_data, ordered_reg):
        """1-std rule: selected fraction must be <= best fraction."""
        X_tr, _, y_tr, _, feat = reg_data
        result = select_by_keep_absolute(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.3, 0.6, 1.0], cv=2)
        assert result['selected_fraction'] <= result['best_fraction'] + 1e-9

    def test_insample_argmax_is_selected(self, reg_data, ordered_reg):
        """For in-sample criteria std=0, so 1-std rule == argmax."""
        X_tr, _, y_tr, _, feat = reg_data
        steps = [0.25, 0.5, 0.75, 1.0]
        result = select_by_keep_absolute(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=steps, cv=99, scoring='bic')
        best_idx  = int(np.argmax(result['scores']))
        assert abs(result['selected_fraction'] - steps[best_idx]) < 1e-9

    @pytest.mark.parametrize("crit", sorted(CRITERION_SCORERS))
    def test_all_scoring_names_work(self, reg_data, ordered_reg, crit):
        X_tr, _, y_tr, _, feat = reg_data
        result = select_by_keep_absolute(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.25, 0.5, 0.75, 1.0], cv=2,
            scoring=crit)
        assert set(result['selected_features']).issubset(set(feat))


# ---------------------------------------------------------------------------
# TestLogTransform  — _log_transform_scores helper
# ---------------------------------------------------------------------------

class TestLogTransform:
    def test_output_nonnegative(self):
        scores = np.array([-500.0, -300.0, -100.0, -50.0])
        out = _core._log_transform_scores(scores)
        assert np.all(out >= 0)

    def test_minimum_is_zero(self):
        scores = np.array([-500.0, -300.0, -100.0])
        out = _core._log_transform_scores(scores)
        assert abs(out.min()) < 1e-12

    def test_monotone_with_original(self):
        """Rank order must be preserved."""
        scores = np.array([-1000.0, -700.0, -400.0, -200.0, -100.0])
        out = _core._log_transform_scores(scores)
        assert list(np.argsort(out)) == list(np.argsort(scores))

    def test_all_finite(self):
        scores = np.array([-1e6, -1e3, -1.0, 0.0, 1.0])
        out = _core._log_transform_scores(scores)
        assert np.all(np.isfinite(out))

    def test_flat_curve_all_zero(self):
        """All-equal scores → all transform outputs are 0."""
        scores = np.array([42.0, 42.0, 42.0])
        out = _core._log_transform_scores(scores)
        assert np.allclose(out, 0.0)

    def test_positive_inputs_also_work(self):
        """Transform is valid for any real-valued scores, not only negative IC."""
        scores = np.array([0.5, 0.7, 0.8, 0.85, 0.9])
        out = _core._log_transform_scores(scores)
        assert out.min() == pytest.approx(0.0)
        assert list(np.argsort(out)) == list(np.argsort(scores))


# ---------------------------------------------------------------------------
# TestSelectByKneeDetection
# ---------------------------------------------------------------------------

kneeliverse = pytest.importorskip("kneeliverse", reason="kneeliverse not installed")


class TestSelectByKneeDetection:
    def test_returns_expected_keys(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = select_by_knee_detection(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2)
        for key in ('knee_fraction', 'selected_fraction', 'selected_features',
                    'method', 'curve_shape', 'curve_direction',
                    'scores', 'std', 'fractions'):
            assert key in result, f"missing key: {key}"

    def test_selected_is_subset(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = select_by_knee_detection(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2)
        assert set(result['selected_features']).issubset(set(feat))

    def test_selected_equals_knee(self, reg_data, ordered_reg):
        """selected_fraction must equal knee_fraction."""
        X_tr, _, y_tr, _, feat = reg_data
        result = select_by_knee_detection(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2)
        assert abs(result['knee_fraction'] - result['selected_fraction']) < 1e-9

    def test_curve_shape_valid(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = select_by_knee_detection(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2)
        assert result['curve_shape']    in ('concave', 'convex', 'unknown')
        assert result['curve_direction'] in ('increasing', 'decreasing', 'unknown')

    @pytest.mark.parametrize("method", KNEE_METHODS)
    def test_all_methods_run(self, reg_data, ordered_reg, method):
        X_tr, _, y_tr, _, feat = reg_data
        result = select_by_knee_detection(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2,
            method=method)
        assert len(result['selected_features']) >= 1

    def test_insample_std_zero_in_knee(self, reg_data, ordered_reg):
        """In-sample criteria in knee detection must also have std=0."""
        X_tr, _, y_tr, _, feat = reg_data
        result = select_by_knee_detection(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=99,
            scoring='bic')
        assert np.allclose(result['std'], 0.0)
        assert not np.any(np.isnan(result['scores']))

    def test_invalid_method_raises(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        with pytest.raises(ValueError, match="bogus"):
            select_by_knee_detection(
                LinearRegression, X_tr, y_tr, feat, ordered_reg,
                task='regression', steps=[0.5, 1.0], cv=2, method='bogus')

    def test_flat_curve_warns_and_falls_back(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        flat_result = {
            'fractions': [0.2, 0.4, 0.6, 0.8, 1.0],
            'scores':    np.ones(5) * 0.5,
            'std':       np.zeros(5),
            'auc':       0.5,
        }
        from unittest.mock import patch
        with patch.object(_core, 'keep_absolute', return_value=flat_result):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = select_by_knee_detection(
                    LinearRegression, X_tr, y_tr, feat, ordered_reg,
                    task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2)
                assert any(issubclass(x.category, UserWarning) for x in w)
        assert len(result['selected_features']) >= 1

    def test_log_transform_result_key_present(self, reg_data, ordered_reg):
        """log_transform_scores key must always be present in result."""
        X_tr, _, y_tr, _, feat = reg_data
        r_false = select_by_knee_detection(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2,
            log_transform_scores=False)
        assert 'log_transform_scores' in r_false
        assert r_false['log_transform_scores'] is False

        r_true = select_by_knee_detection(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2,
            log_transform_scores=True)
        assert r_true['log_transform_scores'] is True

    def test_log_transform_raw_scores_unchanged(self, reg_data, ordered_reg):
        """Raw IC scores in result['scores'] must not be modified by log transform."""
        X_tr, _, y_tr, _, feat = reg_data
        r_false = select_by_knee_detection(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2,
            scoring='bic', log_transform_scores=False)
        r_true = select_by_knee_detection(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2,
            scoring='bic', log_transform_scores=True)
        np.testing.assert_allclose(r_false['scores'], r_true['scores'],
                                   err_msg="log_transform must not alter stored scores")

    def test_log_transform_produces_valid_selection(self, reg_data, ordered_reg):
        """With log_transform=True the knee algorithm must still return a valid subset."""
        X_tr, _, y_tr, _, feat = reg_data
        result = select_by_knee_detection(
            LinearRegression, X_tr, y_tr, feat, ordered_reg,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2,
            scoring='bic', log_transform_scores=True)
        assert set(result['selected_features']).issubset(set(feat))
        assert len(result['selected_features']) >= 1
        assert 0.0 < result['knee_fraction'] <= 1.0


# ---------------------------------------------------------------------------
# TestComputeCriterion
# ---------------------------------------------------------------------------

class TestComputeCriterion:
    @pytest.fixture
    def subset(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        return X_tr, y_tr, feat, list(ordered_reg[:4])

    @pytest.mark.parametrize("crit", ['bic', 'aic', 'sic', 'ebic', 'rebic', 'llf'])
    def test_returns_finite(self, subset, crit):
        X, y, feat, sel = subset
        score, llf, k, n = compute_criterion(
            LinearRegression, X, y, feat, sel, task='regression', criterion=crit)
        assert np.isfinite(score), f"non-finite score for {crit}"
        assert np.isfinite(llf)
        assert k >= 1
        assert n == len(y)

    def test_bic_equals_sic(self, subset):
        X, y, feat, sel = subset
        bic, _, _, _ = compute_criterion(LinearRegression, X, y, feat, sel,
                                         task='regression', criterion='bic')
        sic, _, _, _ = compute_criterion(LinearRegression, X, y, feat, sel,
                                         task='regression', criterion='sic')
        assert abs(bic - sic) < 1e-9

    def test_ebic_le_bic_when_gamma_positive(self, subset):
        """-EBIC <= -BIC when gamma > 0 (extra penalty reduces the score)."""
        X, y, feat, sel = subset
        bic,  _, _, _ = compute_criterion(LinearRegression, X, y, feat, sel,
                                          task='regression', criterion='bic')
        ebic, _, _, _ = compute_criterion(LinearRegression, X, y, feat, sel,
                                          task='regression', criterion='ebic', gamma=1.0)
        assert ebic <= bic + 1e-9

    def test_ebic_gamma0_equals_bic(self, subset):
        """EBIC with gamma=0 must equal BIC (penalty term vanishes)."""
        X, y, feat, sel = subset
        bic,  _, _, _ = compute_criterion(LinearRegression, X, y, feat, sel,
                                          task='regression', criterion='bic')
        ebic, _, _, _ = compute_criterion(LinearRegression, X, y, feat, sel,
                                          task='regression', criterion='ebic', gamma=0.0)
        assert abs(bic - ebic) < 1e-6

    def test_llf_score_equals_log_lik(self, subset):
        """llf score must equal log_lik (no penalty applied)."""
        X, y, feat, sel = subset
        score, llf, _, _ = compute_criterion(LinearRegression, X, y, feat, sel,
                                              task='regression', criterion='llf')
        assert abs(score - llf) < 1e-9

    def test_more_features_higher_llf(self, reg_data, ordered_reg):
        """More features in-sample must give >= log-likelihood (monotone property)."""
        X_tr, _, y_tr, _, feat = reg_data
        small = list(ordered_reg[:2])
        large = list(ordered_reg[:6])
        s_sm, _, _, _ = compute_criterion(LinearRegression, X_tr, y_tr, feat, small,
                                           task='regression', criterion='llf')
        s_lg, _, _, _ = compute_criterion(LinearRegression, X_tr, y_tr, feat, large,
                                           task='regression', criterion='llf')
        assert s_lg >= s_sm - 1e-6

    def test_invalid_criterion_raises(self, subset):
        X, y, feat, sel = subset
        with pytest.raises(ValueError, match="bad_crit"):
            compute_criterion(LinearRegression, X, y, feat, sel,
                              task='regression', criterion='bad_crit')

    def test_classification_bic(self, cls_data):
        from sklearn.linear_model import LogisticRegression
        X_tr, _, y_tr, _, feat = cls_data
        from shap_selection import shap_select as ss
        m = LogisticRegression(max_iter=500)
        m.fit(X_tr, y_tr)
        names, _ = ss(m, X_tr, feat, task='classification')
        # Pass a configured instance — no lambda required
        score, llf, k, n = compute_criterion(
            LogisticRegression(max_iter=500),
            X_tr, y_tr, feat, list(names[:4]),
            task='classification', criterion='bic')
        assert np.isfinite(score)

    def test_accepts_numpy_array_feature_names(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        sel = ordered_reg[:3]   # numpy array
        score, _, _, _ = compute_criterion(LinearRegression, X_tr, y_tr, feat, sel,
                                            task='regression', criterion='bic')
        assert np.isfinite(score)


# ---------------------------------------------------------------------------
# TestAutoSelect
# ---------------------------------------------------------------------------

class TestAutoSelect:
    # ------------------------------------------------------------------ #
    # Unified mode  (criterion=None, default)                             #
    # Both legs share the same scoring= and cv=                           #
    # ------------------------------------------------------------------ #

    def test_unified_returns_expected_keys(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring='r2')
        for key in ('selected_features', 'winner', 'scoring', 'criterion',
                    'absolute_features', 'knee_features',
                    'absolute_score', 'knee_score',
                    'absolute_result', 'knee_result'):
            assert key in result, f"missing key: {key}"
        # Split-mode keys must NOT be present in unified mode
        assert 'absolute_criterion_score' not in result
        assert 'knee_criterion_score'     not in result

    def test_unified_criterion_key_is_none(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.5, 1.0], cv=2)
        assert result['criterion'] is None

    def test_unified_winner_valid(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring='r2')
        assert result['winner'] in ('absolute', 'knee')

    def test_unified_selected_matches_winner(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring='r2')
        expected = result[f"{result['winner']}_result"]['selected_features']
        np.testing.assert_array_equal(result['selected_features'], expected)

    def test_unified_winner_has_higher_score(self, reg_data, ordered_reg):
        """In unified mode the winner's sweep score must be >= the loser's."""
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring='r2')
        if result['winner'] == 'absolute':
            assert result['absolute_score'] >= result['knee_score'] - 1e-9
        else:
            assert result['knee_score'] >= result['absolute_score'] - 1e-9

    @pytest.mark.parametrize("scoring", ['llf', 'r2'])
    def test_unified_cv_scorers(self, reg_data, ordered_reg, scoring):
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring=scoring)
        assert set(result['selected_features']).issubset(set(feat))

    @pytest.mark.parametrize("scoring", ['bic', 'aic', 'sic', 'ebic'])
    def test_unified_insample_scorers(self, reg_data, ordered_reg, scoring):
        """Passing an IC as scoring= still works (unified mode, both legs use IC)."""
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring=scoring)
        assert set(result['selected_features']).issubset(set(feat))

    def test_unified_default_scoring(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.5, 1.0], cv=2)
        assert set(result['selected_features']).issubset(set(feat))

    # ------------------------------------------------------------------ #
    # Split mode  (criterion != None)                                     #
    # keep_absolute  <- scoring + cv  (CV path)                           #
    # knee_detection <- criterion     (in-sample, cv forced None)         #
    # winner         <- IC comparison on both selected subsets            #
    # ------------------------------------------------------------------ #

    def test_split_returns_expected_keys(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring='r2', criterion='bic')
        for key in ('selected_features', 'winner', 'scoring', 'criterion',
                    'absolute_features', 'knee_features',
                    'absolute_score', 'knee_score',
                    'absolute_result', 'knee_result',
                    'absolute_criterion_score', 'knee_criterion_score',
                    'criterion_gamma'):
            assert key in result, f"missing key: {key}"

    def test_split_criterion_key_preserved(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.5, 1.0],
                             cv=2, scoring='r2', criterion='bic')
        assert result['criterion'] == 'bic'
        assert result['scoring']   == 'r2'

    def test_split_absolute_uses_cv_scorer(self, reg_data, ordered_reg):
        """keep_absolute leg must use scoring='r2' through cross_val_score."""
        from unittest.mock import patch
        X_tr, _, y_tr, _, feat = reg_data
        captured = {}
        orig = _core.cross_val_score

        def mock_cvs(est, X, y, cv=None, scoring=None, **kw):
            captured.setdefault('scorings', set()).add(
                scoring if isinstance(scoring, str) else '<callable>'
            )
            captured['cv'] = cv
            return orig(est, X, y, cv=cv, scoring=scoring, **kw)

        with patch.object(_core, 'cross_val_score', mock_cvs):
            auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                        task='regression', steps=[0.5, 1.0],
                        cv=3, scoring='r2', criterion='bic')

        assert 'r2' in captured.get('scorings', set()), \
            "keep_absolute should use scoring='r2' via cross_val_score"
        assert captured.get('cv') == 3, "cv should be forwarded to keep_absolute"

    def test_split_knee_uses_insample(self, reg_data, ordered_reg):
        """knee_detection leg must use criterion='bic', NOT cross_val_score."""
        X_tr, _, y_tr, _, feat = reg_data
        # We verify that the knee sweep scores have std=0 (in-sample path).
        # cv=3 (not 99) to avoid NaN from rank-deficient CV folds on small data.
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=3, scoring='r2', criterion='bic')
        knee_std = result['knee_result']['std']
        assert np.allclose(knee_std, 0.0), \
            "knee_detection std must be 0 when criterion is set (in-sample path)"

    def test_split_absolute_sweep_is_cv(self, reg_data, ordered_reg):
        """keep_absolute sweep std must be non-trivially zero (real CV folds)."""
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=3, scoring='r2', criterion='bic')
        abs_std = result['absolute_result']['std']
        # std comes from real CV folds — all-zero would be pathological
        assert np.all(abs_std >= 0)
        assert np.all(np.isfinite(abs_std))

    def test_split_winner_decided_by_criterion(self, reg_data, ordered_reg):
        """Winner is the subset with the higher IC score (absolute_criterion_score
        or knee_criterion_score), not the raw sweep score."""
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring='r2', criterion='bic')
        if result['winner'] == 'absolute':
            assert result['absolute_criterion_score'] >= \
                   result['knee_criterion_score'] - 1e-9
        else:
            assert result['knee_criterion_score'] >= \
                   result['absolute_criterion_score'] - 1e-9

    def test_split_criterion_scores_finite(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring='r2', criterion='bic')
        assert np.isfinite(result['absolute_criterion_score'])
        assert np.isfinite(result['knee_criterion_score'])

    def test_split_selected_matches_winner(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring='r2', criterion='bic')
        expected = result[f"{result['winner']}_result"]['selected_features']
        np.testing.assert_array_equal(result['selected_features'], expected)

    @pytest.mark.parametrize("crit", ['bic', 'aic', 'ebic'])
    def test_split_all_criterion_values(self, reg_data, ordered_reg, crit):
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring='r2', criterion=crit)
        assert set(result['selected_features']).issubset(set(feat))

    def test_split_criterion_tuple_gamma(self, reg_data, ordered_reg):
        """(criterion, gamma) tuple must work for the criterion arg."""
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.5, 1.0],
                             cv=2, scoring='r2', criterion=('ebic', 0.3))
        assert set(result['selected_features']).issubset(set(feat))
        assert abs(result['criterion_gamma'] - 0.3) < 1e-9

    def test_split_invalid_criterion_raises(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        with pytest.raises(ValueError):
            auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                        task='regression', steps=[0.5, 1.0],
                        cv=2, scoring='r2', criterion='llf')  # llf not in-sample

    def test_split_gamma_default_is_1(self, reg_data, ordered_reg):
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.5, 1.0],
                             cv=2, scoring='r2', criterion='ebic')
        assert result['criterion_gamma'] == 1.0

    # ------------------------------------------------------------------ #
    # log_transform_scores — forwarded to knee leg only                   #
    # ------------------------------------------------------------------ #

    def test_log_transform_forwarded_to_knee(self, reg_data, ordered_reg):
        """log_transform_scores=True must reach knee_result['log_transform_scores']."""
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring='r2', criterion='bic',
                             log_transform_scores=True)
        assert result['knee_result']['log_transform_scores'] is True

    def test_log_transform_not_applied_to_absolute(self, reg_data, ordered_reg):
        """log_transform_scores must NOT affect the keep_absolute leg."""
        X_tr, _, y_tr, _, feat = reg_data
        # keep_absolute leg: run with and without log_transform, compare scores
        r_false = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                              task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                              cv=2, scoring='r2', criterion='bic',
                              log_transform_scores=False)
        r_true  = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                              task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                              cv=2, scoring='r2', criterion='bic',
                              log_transform_scores=True)
        np.testing.assert_allclose(
            r_false['absolute_result']['scores'],
            r_true['absolute_result']['scores'],
            err_msg="log_transform_scores must not alter keep_absolute scores",
        )

    def test_log_transform_unified_mode(self, reg_data, ordered_reg):
        """log_transform_scores=True with criterion=None (unified mode) works."""
        X_tr, _, y_tr, _, feat = reg_data
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=2, scoring='r2', log_transform_scores=True)
        assert set(result['selected_features']).issubset(set(feat))
        assert result['knee_result']['log_transform_scores'] is True


# ---------------------------------------------------------------------------
# TestShapThresholdSelect
# ---------------------------------------------------------------------------

class TestShapThresholdSelect:
    def test_top_k(self, fitted_linear, reg_data):
        X_tr, _, _, _, feat = reg_data
        sel, _, _, _ = shap_threshold_select(
            fitted_linear, X_tr, feat, task='regression', top_k=3)
        assert len(sel) == 3

    def test_threshold(self, fitted_linear, reg_data):
        X_tr, _, _, _, feat = reg_data
        _, all_imp = shap_select(fitted_linear, X_tr, feat, task='regression')
        cutoff = float(np.median(all_imp))
        sel, sel_imp, _, _ = shap_threshold_select(
            fitted_linear, X_tr, feat, task='regression', threshold=cutoff)
        assert np.all(sel_imp >= cutoff)

    def test_no_filter_returns_all(self, fitted_linear, reg_data):
        X_tr, _, _, _, feat = reg_data
        sel, _, _, _ = shap_threshold_select(
            fitted_linear, X_tr, feat, task='regression')
        assert len(sel) == len(feat)

    def test_raises_on_both_params(self, fitted_linear, reg_data):
        X_tr, _, _, _, feat = reg_data
        with pytest.raises(ValueError):
            shap_threshold_select(
                fitted_linear, X_tr, feat,
                task='regression', top_k=2, threshold=1.0)


# ---------------------------------------------------------------------------
# TestApplyFeatureSelection
# ---------------------------------------------------------------------------

class TestApplyFeatureSelection:
    def test_basic_numpy(self):
        X = np.arange(12).reshape(4, 3).astype(float)
        r = apply_feature_selection(X, ['a','b','c'], ['a','c'])
        assert r.shape == (4, 2)
        np.testing.assert_array_equal(r[:, 0], X[:, 0])
        np.testing.assert_array_equal(r[:, 1], X[:, 2])

    def test_single_feature(self):
        X = np.arange(12).reshape(4, 3).astype(float)
        r = apply_feature_selection(X, ['a','b','c'], ['b'])
        assert r.shape == (4, 1)

    def test_numpy_array_names(self):
        X = np.arange(12).reshape(4, 3).astype(float)
        r = apply_feature_selection(X, np.array(['a','b','c']), np.array(['a','c']))
        assert r.shape == (4, 2)

    def test_pandas_dataframe(self):
        pytest.importorskip("pandas")
        import pandas as pd
        df = pd.DataFrame({'a':[1,2],'b':[3,4],'c':[5,6]})
        r  = apply_feature_selection(df, ['a','b','c'], ['a','c'])
        assert list(r.columns) == ['a','c']

    def test_unknown_feature_raises(self):
        X = np.zeros((4, 3))
        with pytest.raises(ValueError):
            apply_feature_selection(X, ['a','b','c'], ['a','z'])

    def test_preserves_order(self):
        X = np.arange(8).reshape(2, 4).astype(float)
        r = apply_feature_selection(X, ['a','b','c','d'], ['d','a'])
        np.testing.assert_array_equal(r[:, 0], X[:, 3])  # d
        np.testing.assert_array_equal(r[:, 1], X[:, 0])  # a


# ---------------------------------------------------------------------------
# TestScoringDispatch  — _parse_scoring routing
# ---------------------------------------------------------------------------

class TestScoringDispatch:
    def test_sklearn_string_goes_cv(self):
        cv, ic, gamma = _core._parse_scoring('r2', 'regression')
        assert cv == 'r2'
        assert ic is None

    def test_llf_goes_cv(self):
        cv, ic, gamma = _core._parse_scoring('llf', 'regression')
        assert callable(cv)
        assert ic is None

    def test_bic_goes_insample(self):
        cv, ic, gamma = _core._parse_scoring('bic', 'regression')
        assert cv is None
        assert ic == 'bic'
        assert gamma == 1.0

    @pytest.mark.parametrize("crit", sorted(INSAMPLE_CRITERIA))
    def test_all_insample_criteria_routed(self, crit):
        cv, ic, gamma = _core._parse_scoring(crit, 'regression')
        assert cv is None
        assert ic == crit.lower()

    def test_tuple_sets_gamma(self):
        cv, ic, gamma = _core._parse_scoring(('ebic', 0.3), 'regression')
        assert cv is None
        assert ic == 'ebic'
        assert abs(gamma - 0.3) < 1e-9

    def test_invalid_tuple_raises(self):
        with pytest.raises(ValueError):
            _core._parse_scoring(('llf', 0.5), 'regression')  # llf is not in-sample

    def test_none_goes_cv(self):
        cv, ic, gamma = _core._parse_scoring(None, 'regression')
        assert cv is None   # None passed through; default applied later
        assert ic is None

    def test_callable_goes_cv(self):
        fn = lambda est, X, y: 0.0
        cv, ic, gamma = _core._parse_scoring(fn, 'regression')
        assert cv is fn
        assert ic is None


# ---------------------------------------------------------------------------
# TestDefaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_cv_is_3(self, reg_data, ordered_reg):
        from unittest.mock import patch
        X_tr, _, y_tr, _, feat = reg_data
        captured = {}
        orig = _core.cross_val_score

        def mock_cvs(est, X, y, cv=None, scoring=None, **kw):
            captured['cv'] = cv
            return orig(est, X, y, cv=cv, scoring=scoring, **kw)

        with patch.object(_core, 'cross_val_score', mock_cvs):
            keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                          task='regression', steps=[1.0])
        assert captured['cv'] == 3

    def test_default_scoring_regression_r2(self, reg_data, ordered_reg):
        from unittest.mock import patch
        X_tr, _, y_tr, _, feat = reg_data
        captured = {}
        orig = _core.cross_val_score

        def mock_cvs(est, X, y, cv=None, scoring=None, **kw):
            captured['scoring'] = scoring
            return orig(est, X, y, cv=cv, scoring=scoring, **kw)

        with patch.object(_core, 'cross_val_score', mock_cvs):
            keep_absolute(LinearRegression, X_tr, y_tr, feat, ordered_reg,
                          task='regression', steps=[1.0])
        assert captured['scoring'] == 'r2'

    def test_default_scoring_classification_f1(self, cls_data):
        from unittest.mock import patch
        from sklearn.linear_model import LogisticRegression
        X_tr, _, y_tr, _, feat = cls_data
        from shap_selection import shap_select as ss
        m = LogisticRegression(max_iter=500).fit(X_tr, y_tr)
        names, _ = ss(m, X_tr, feat, task='classification')
        captured = {}
        orig = _core.cross_val_score

        def mock_cvs(est, X, y, cv=None, scoring=None, **kw):
            captured['scoring'] = scoring
            return orig(est, X, y, cv=cv, scoring=scoring, **kw)

        with patch.object(_core, 'cross_val_score', mock_cvs):
            keep_absolute(lambda: LogisticRegression(max_iter=500),
                          X_tr, y_tr, feat, names,
                          task='classification', steps=[1.0])
        assert captured['scoring'] == 'f1_weighted'

    def test_knee_methods_list(self):
        assert set(KNEE_METHODS) == {'kneedle', 'dfdt', 'curvature', 'menger', 'lmethod'}

    def test_insample_criteria_set(self):
        assert INSAMPLE_CRITERIA == frozenset({'bic', 'aic', 'sic', 'ebic', 'rebic'})

    def test_criterion_scorers_superset(self):
        assert INSAMPLE_CRITERIA.issubset(CRITERION_SCORERS)
        assert 'llf' in CRITERION_SCORERS


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_feature(self):
        rng = np.random.default_rng(0)
        X   = rng.standard_normal((50, 1))
        y   = X[:, 0] * 2 + 0.1
        feat = np.array(['x0'])
        m    = LinearRegression().fit(X, y)
        names, _ = shap_select(m, X, feat, task='regression')
        assert len(names) == 1
        result = select_by_keep_absolute(
            LinearRegression, X, y, feat, names, task='regression', cv=2)
        assert len(result['selected_features']) == 1

    def test_single_feature_insample(self):
        rng  = np.random.default_rng(1)
        X    = rng.standard_normal((50, 1))
        y    = X[:, 0] * 3 + 0.2
        feat = np.array(['x0'])
        m    = LinearRegression().fit(X, y)
        names, _ = shap_select(m, X, feat, task='regression')
        result = select_by_keep_absolute(
            LinearRegression, X, y, feat, names,
            task='regression', scoring='bic')
        assert len(result['selected_features']) == 1
        assert np.allclose(result['std'], 0.0)

    def test_insample_cv_truly_ignored(self):
        """Passing cv=999 with an in-sample criterion must not raise."""
        rng  = np.random.default_rng(2)
        X    = rng.standard_normal((40, 3))
        y    = X[:, 0] + 0.1
        feat = np.array(['a', 'b', 'c'])
        m    = LinearRegression().fit(X, y)
        names, _ = shap_select(m, X, feat, task='regression')
        # cv=999 would cause cross_val_score to fail; this must not be called
        result = select_by_keep_absolute(
            LinearRegression, X, y, feat, names,
            task='regression', cv=999, scoring='bic')
        assert len(result['selected_features']) >= 1


# ---------------------------------------------------------------------------
# New tests: EBIC/REBIC overflow, 3D SHAP arrays, REBIC formula, consistency
# ---------------------------------------------------------------------------

class TestEbicHighDimensional:
    """Test that EBIC/REBIC do not overflow with many features."""

    @pytest.fixture(scope="class")
    def high_dim_data(self):
        rng = np.random.default_rng(99)
        n, p = 300, 150
        X = rng.standard_normal((n, p))
        y = X[:, 0] * 3.0 + X[:, 1] * 1.5 + rng.standard_normal(n) * 0.1
        feat = np.array([f"f{i}" for i in range(p)])
        return X, y, feat

    def test_ebic_no_overflow(self, high_dim_data):
        """EBIC with 150 features must not raise OverflowError."""
        X, y, feat = high_dim_data
        selected = list(feat[:10])
        score, log_lik, k, n = compute_criterion(
            LinearRegression, X, y, feat, selected,
            task='regression', criterion='ebic', gamma=1.0,
        )
        assert np.isfinite(score)
        assert np.isfinite(log_lik)

    def test_rebic_no_overflow(self, high_dim_data):
        """REBIC with 150 features must not raise OverflowError."""
        X, y, feat = high_dim_data
        selected = list(feat[:10])
        score, log_lik, k, n = compute_criterion(
            LinearRegression, X, y, feat, selected,
            task='regression', criterion='rebic', gamma=1.0,
        )
        assert np.isfinite(score)
        assert np.isfinite(log_lik)

    def test_ebic_gamma0_still_equals_bic_high_dim(self, high_dim_data):
        """EBIC(gamma=0) must equal BIC even at high dimensions."""
        X, y, feat = high_dim_data
        selected = list(feat[:20])
        ebic_score, _, _, _ = compute_criterion(
            LinearRegression, X, y, feat, selected,
            task='regression', criterion='ebic', gamma=0.0,
        )
        bic_score, _, _, _ = compute_criterion(
            LinearRegression, X, y, feat, selected,
            task='regression', criterion='bic',
        )
        assert abs(ebic_score - bic_score) < 1e-6

    def test_ebic_sweep_high_dim(self, high_dim_data):
        """Full keep_absolute sweep with EBIC at p=150 must complete."""
        X, y, feat = high_dim_data
        m = LinearRegression().fit(X, y)
        ordered, _ = shap_select(m, X, feat, task='regression')
        result = select_by_keep_absolute(
            LinearRegression, X, y, feat, ordered,
            task='regression', scoring='ebic', step_by='fraction',
        )
        assert all(np.isfinite(result['scores']))
        assert len(result['selected_features']) >= 1


class TestShapOrdering3D:
    """Test that _shap_ordering handles both list-of-2D and 3D ndarray SHAP values."""

    def test_3d_ndarray_classification(self):
        """Modern SHAP: 3D ndarray (n_samples, n_features, n_classes)."""
        rng = np.random.default_rng(42)
        n_samples, n_features, n_classes = 50, 5, 3
        feat = np.array([f"f{i}" for i in range(n_features)])
        # Make f0 most important across all classes
        shap_3d = rng.standard_normal((n_samples, n_features, n_classes)) * 0.01
        shap_3d[:, 0, :] = rng.standard_normal((n_samples, n_classes)) * 10.0

        names, importance = _core._shap_ordering(feat, shap_3d, task='classification')
        assert names.shape == (n_features,)
        assert importance.shape == (n_features,)
        assert names[0] == 'f0', "Most important feature should be f0"
        assert np.all(importance[:-1] >= importance[1:]), "Importance must be descending"

    def test_list_of_2d_classification(self):
        """Legacy SHAP: list of 2D arrays [class0_shap, class1_shap, ...]."""
        rng = np.random.default_rng(42)
        n_samples, n_features, n_classes = 50, 5, 3
        feat = np.array([f"f{i}" for i in range(n_features)])
        # Make f1 most important
        shap_list = []
        for _ in range(n_classes):
            arr = rng.standard_normal((n_samples, n_features)) * 0.01
            arr[:, 1] = rng.standard_normal(n_samples) * 10.0
            shap_list.append(arr)

        names, importance = _core._shap_ordering(feat, shap_list, task='classification')
        assert names.shape == (n_features,)
        assert importance.shape == (n_features,)
        assert names[0] == 'f1', "Most important feature should be f1"

    def test_3d_and_list_agree(self):
        """3D ndarray and list-of-2D must produce the same ranking."""
        rng = np.random.default_rng(7)
        n_samples, n_features, n_classes = 80, 6, 4
        feat = np.array([f"f{i}" for i in range(n_features)])

        # Build as list-of-2D first (n_classes, n_samples, n_features)
        shap_list = [rng.standard_normal((n_samples, n_features)) for _ in range(n_classes)]

        # Convert to 3D ndarray (n_samples, n_features, n_classes)
        shap_3d = np.stack(shap_list, axis=-1)  # (n_samples, n_features, n_classes)

        names_list, imp_list = _core._shap_ordering(feat, shap_list, task='classification')
        names_3d, imp_3d = _core._shap_ordering(feat, shap_3d, task='classification')

        np.testing.assert_array_equal(names_list, names_3d)
        np.testing.assert_allclose(imp_list, imp_3d, atol=1e-10)

    def test_2d_regression_unchanged(self):
        """Regression (2D SHAP) must still work correctly."""
        rng = np.random.default_rng(42)
        feat = np.array(['a', 'b', 'c'])
        shap_2d = rng.standard_normal((100, 3))
        shap_2d[:, 2] = rng.standard_normal(100) * 50.0  # c is strongest

        names, importance = _core._shap_ordering(feat, shap_2d, task='regression')
        assert names[0] == 'c'


class TestRebicNullModel:
    """Test that REBIC correctly uses the intercept-only model as the null."""

    def test_rebic_uncentered_target(self):
        """REBIC on uncentered y must differ from ||y||^2-based computation."""
        rng = np.random.default_rng(88)
        X = rng.standard_normal((100, 4))
        y = X[:, 0] * 2.0 + 100.0  # large non-zero mean
        feat = [f"f{i}" for i in range(4)]
        selected = ['f0', 'f1']

        score, log_lik, k, n = compute_criterion(
            LinearRegression, X, y, feat, selected,
            task='regression', criterion='rebic', gamma=1.0,
        )
        assert np.isfinite(score), "REBIC must be finite for uncentered targets"
        # REBIC must be better (higher) with the true feature than with noise only
        score_noise, _, _, _ = compute_criterion(
            LinearRegression, X, y, feat, ['f2', 'f3'],
            task='regression', criterion='rebic', gamma=1.0,
        )
        assert score > score_noise, (
            "REBIC for true features must beat noise features"
        )

    def test_rebic_centered_vs_uncentered(self):
        """REBIC should give different scores for centered vs uncentered y."""
        rng = np.random.default_rng(77)
        X = rng.standard_normal((100, 3))
        feat = ['a', 'b', 'c']
        selected = ['a']

        y_centered = X[:, 0] * 2.0
        y_shifted = y_centered + 500.0  # huge shift

        score_c, _, _, _ = compute_criterion(
            LinearRegression, X, y_centered, feat, selected,
            task='regression', criterion='rebic',
        )
        score_s, _, _, _ = compute_criterion(
            LinearRegression, X, y_shifted, feat, selected,
            task='regression', criterion='rebic',
        )
        # With the corrected formula, these should be the same because
        # Ridge/LR fit_intercept=True absorbs the mean shift.
        # The null-model RSS (y - y_bar)^2 is identical for both.
        # So REBIC should be equal (or very close).
        assert abs(score_c - score_s) < 1.0, (
            "REBIC scores should be similar regardless of target mean shift "
            f"(got {score_c:.2f} vs {score_s:.2f})"
        )


class TestReturnTypeConsistency:
    """Verify that selected_features is always a list across all functions."""

    @pytest.fixture(scope="class")
    def setup(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((80, 5))
        y = X[:, 0] + 0.1 * rng.standard_normal(80)
        feat = [f"f{i}" for i in range(5)]
        m = LinearRegression().fit(X, y)
        ordered, _ = shap_select(m, X, feat, task='regression')
        return X, y, feat, ordered

    def test_select_by_keep_absolute_returns_list(self, setup):
        X, y, feat, ordered = setup
        result = select_by_keep_absolute(
            LinearRegression, X, y, feat, ordered, task='regression')
        assert isinstance(result['selected_features'], list)

    def test_select_by_knee_detection_returns_list(self, setup):
        X, y, feat, ordered = setup
        result = select_by_knee_detection(
            LinearRegression, X, y, feat, ordered,
            task='regression', step_by='feature')
        assert isinstance(result['selected_features'], list)

    def test_auto_select_returns_lists(self, setup):
        X, y, feat, ordered = setup
        result = auto_select(
            LinearRegression, X, y, feat, ordered, task='regression')
        assert isinstance(result['selected_features'], list)
        assert isinstance(result['absolute_features'], list)
        assert isinstance(result['knee_features'], list)


class TestPreExistingBugFix:
    """Regression test: the pre-existing test that fails with cv=99 on small data."""

    def test_split_knee_uses_insample_fixed(self):
        """Same intent as the original test but with cv=3 to avoid NaN."""
        X, y = make_regression(n_samples=200, n_features=8, noise=0.1, random_state=42)
        feat = np.array([f"f{i}" for i in range(8)])
        X_tr = X[:160]
        y_tr = y[:160]
        m = LinearRegression().fit(X_tr, y_tr)
        ordered, _ = shap_select(m, X_tr, feat, task='regression')
        result = auto_select(LinearRegression, X_tr, y_tr, feat, ordered,
                             task='regression', steps=[0.25, 0.5, 0.75, 1.0],
                             cv=3, scoring='r2', criterion='bic')
        knee_std = result['knee_result']['std']
        assert np.allclose(knee_std, 0.0), \
            "knee_detection std must be 0 when criterion is set (in-sample path)"
