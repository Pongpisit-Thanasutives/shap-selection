"""
Tests for shap_selection.

Uses simple synthetic data and sklearn's LinearRegression / DecisionTreeRegressor
so the suite runs quickly without large datasets.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression

from shap_selection import (
    shap_select,
    keep_absolute,
    select_by_keep_absolute,
    shap_threshold_select,
    apply_feature_selection,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=6, noise=0.1, random_state=42)
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    split = 160
    return X[:split], X[split:], y[:split], y[split:], feature_names


@pytest.fixture
def fitted_linear(regression_data):
    X_train, _, y_train, _, _ = regression_data
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def fitted_tree(regression_data):
    X_train, _, y_train, _, _ = regression_data
    model = DecisionTreeRegressor(max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    return model


# ---------------------------------------------------------------------------
# shap_select
# ---------------------------------------------------------------------------

class TestShapSelect:
    def test_returns_all_features(self, fitted_linear, regression_data):
        X_train, _, _, _, feature_names = regression_data
        names, importance = shap_select(fitted_linear, X_train, feature_names, task='regression')
        assert len(names) == len(feature_names)
        assert len(importance) == len(feature_names)

    def test_sorted_descending(self, fitted_linear, regression_data):
        X_train, _, _, _, feature_names = regression_data
        _, importance = shap_select(fitted_linear, X_train, feature_names, task='regression')
        assert list(importance) == sorted(importance, reverse=True)

    def test_importance_nonnegative(self, fitted_linear, regression_data):
        X_train, _, _, _, feature_names = regression_data
        _, importance = shap_select(fitted_linear, X_train, feature_names, task='regression')
        assert np.all(importance >= 0)

    def test_feature_names_are_subset(self, fitted_linear, regression_data):
        X_train, _, _, _, feature_names = regression_data
        names, _ = shap_select(fitted_linear, X_train, feature_names, task='regression')
        assert set(names) == set(feature_names)

    def test_works_with_tree_model(self, fitted_tree, regression_data):
        X_train, _, _, _, feature_names = regression_data
        names, importance = shap_select(fitted_tree, X_train, feature_names, task='regression')
        assert len(names) == len(feature_names)
        assert list(importance) == sorted(importance, reverse=True)

    def test_works_with_ridge(self, regression_data):
        X_train, _, y_train, _, feature_names = regression_data
        model = Ridge()
        model.fit(X_train, y_train)
        names, importance = shap_select(model, X_train, feature_names, task='regression')
        assert len(names) == len(feature_names)

    def test_same_array_for_train_and_test(self, fitted_linear, regression_data):
        """No X_test needed; shap_select uses X_train for both background and explanation."""
        X_train, _, _, _, feature_names = regression_data
        names, importance = shap_select(
            fitted_linear, X_train, feature_names, task='regression'
        )
        assert len(names) == len(feature_names)
        assert list(importance) == sorted(importance, reverse=True)


# ---------------------------------------------------------------------------
# keep_absolute
# ---------------------------------------------------------------------------

class TestKeepAbsolute:
    def test_returns_expected_keys(self, regression_data):
        X_train, _, y_train, _, feature_names = regression_data
        model = LinearRegression()
        model.fit(X_train, y_train)
        names, _ = shap_select(model, X_train, feature_names, task='regression')

        result = keep_absolute(
            LinearRegression, X_train, y_train, feature_names, names,
            task='regression', steps=[0.5, 1.0], cv=2,
        )
        assert set(result.keys()) == {'fractions', 'scores', 'std', 'auc'}

    def test_score_length_matches_steps(self, regression_data):
        X_train, _, y_train, _, feature_names = regression_data
        model = LinearRegression()
        model.fit(X_train, y_train)
        names, _ = shap_select(model, X_train, feature_names, task='regression')

        steps = [0.3, 0.6, 1.0]
        result = keep_absolute(
            LinearRegression, X_train, y_train, feature_names, names,
            task='regression', steps=steps, cv=2,
        )
        assert len(result['scores']) == len(steps)
        assert len(result['std']) == len(steps)

    def test_auc_is_scalar(self, regression_data):
        X_train, _, y_train, _, feature_names = regression_data
        model = LinearRegression()
        model.fit(X_train, y_train)
        names, _ = shap_select(model, X_train, feature_names, task='regression')

        result = keep_absolute(
            LinearRegression, X_train, y_train, feature_names, names,
            task='regression', steps=[0.5, 1.0], cv=2,
        )
        assert isinstance(result['auc'], float)


# ---------------------------------------------------------------------------
# select_by_keep_absolute
# ---------------------------------------------------------------------------

class TestSelectByKeepAbsolute:
    def test_returns_expected_keys(self, regression_data):
        X_train, _, y_train, _, feature_names = regression_data
        model = LinearRegression()
        model.fit(X_train, y_train)
        names, _ = shap_select(model, X_train, feature_names, task='regression')

        result = select_by_keep_absolute(
            LinearRegression, X_train, y_train, feature_names, names,
            task='regression', steps=[0.5, 1.0], cv=2,
        )
        assert 'selected_features' in result
        assert 'selected_fraction' in result
        assert 'best_fraction' in result

    def test_selected_features_are_subset(self, regression_data):
        X_train, _, y_train, _, feature_names = regression_data
        model = LinearRegression()
        model.fit(X_train, y_train)
        names, _ = shap_select(model, X_train, feature_names, task='regression')

        result = select_by_keep_absolute(
            LinearRegression, X_train, y_train, feature_names, names,
            task='regression', steps=[0.5, 1.0], cv=2,
        )
        assert set(result['selected_features']).issubset(set(feature_names))
        assert len(result['selected_features']) >= 1

    def test_selected_fraction_lte_best(self, regression_data):
        X_train, _, y_train, _, feature_names = regression_data
        model = LinearRegression()
        model.fit(X_train, y_train)
        names, _ = shap_select(model, X_train, feature_names, task='regression')

        result = select_by_keep_absolute(
            LinearRegression, X_train, y_train, feature_names, names,
            task='regression', steps=[0.3, 0.6, 1.0], cv=2,
        )
        assert result['selected_fraction'] <= result['best_fraction']


# ---------------------------------------------------------------------------
# shap_threshold_select
# ---------------------------------------------------------------------------

class TestShapThresholdSelect:
    def test_top_k(self, fitted_linear, regression_data):
        X_train, _, _, _, feature_names = regression_data
        sel, sel_imp, all_names, all_imp = shap_threshold_select(
            fitted_linear, X_train, feature_names,
            task='regression', top_k=3,
        )
        assert len(sel) == 3

    def test_threshold(self, fitted_linear, regression_data):
        X_train, _, _, _, feature_names = regression_data
        _, all_imp = shap_select(fitted_linear, X_train, feature_names, task='regression')
        cutoff = float(np.median(all_imp))

        sel, sel_imp, _, _ = shap_threshold_select(
            fitted_linear, X_train, feature_names,
            task='regression', threshold=cutoff,
        )
        assert np.all(sel_imp >= cutoff)

    def test_no_filter_returns_all(self, fitted_linear, regression_data):
        X_train, _, _, _, feature_names = regression_data
        sel, _, _, _ = shap_threshold_select(
            fitted_linear, X_train, feature_names, task='regression',
        )
        assert len(sel) == len(feature_names)

    def test_raises_on_both_params(self, fitted_linear, regression_data):
        X_train, _, _, _, feature_names = regression_data
        with pytest.raises(ValueError):
            shap_threshold_select(
                fitted_linear, X_train, feature_names,
                task='regression', top_k=2, threshold=1.0,
            )


# ---------------------------------------------------------------------------
# apply_feature_selection
# ---------------------------------------------------------------------------

class TestApplyFeatureSelection:
    def test_numpy_array(self):
        X = np.arange(12).reshape(4, 3).astype(float)
        names = ['a', 'b', 'c']
        result = apply_feature_selection(X, names, ['a', 'c'])
        assert result.shape == (4, 2)
        np.testing.assert_array_equal(result[:, 0], X[:, 0])
        np.testing.assert_array_equal(result[:, 1], X[:, 2])

    def test_single_feature(self):
        X = np.arange(12).reshape(4, 3).astype(float)
        names = ['a', 'b', 'c']
        result = apply_feature_selection(X, names, ['b'])
        assert result.shape == (4, 1)

    def test_pandas_dataframe(self):
        pytest.importorskip("pandas")
        import pandas as pd
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})
        result = apply_feature_selection(df, ['a', 'b', 'c'], ['a', 'c'])
        assert list(result.columns) == ['a', 'c']

    def test_unknown_feature_raises(self):
        X = np.zeros((4, 3))
        with pytest.raises(ValueError):
            apply_feature_selection(X, ['a', 'b', 'c'], ['a', 'z'])


# ---------------------------------------------------------------------------
# select_by_knee_detection
# ---------------------------------------------------------------------------

class TestSelectByKneeDetection:
    def test_requires_kneed(self, regression_data, monkeypatch):
        """Should raise ImportError with helpful message if kneed is missing."""
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == 'kneed':
                raise ImportError("No module named 'kneed'")
            return real_import(name, *args, **kwargs)

        kneed = pytest.importorskip("kneed")
        monkeypatch.setattr(builtins, "__import__", mock_import)

        X_train, _, y_train, _, feature_names = regression_data
        model = LinearRegression()
        model.fit(X_train, y_train)
        names, _ = shap_select(model, X_train, feature_names, task='regression')

        from shap_selection import select_by_knee_detection
        with pytest.raises(ImportError, match="kneed"):
            select_by_knee_detection(
                LinearRegression, X_train, y_train, feature_names, names,
                task='regression', steps=[0.5, 1.0], cv=2,
            )

    def test_returns_expected_keys(self, regression_data):
        pytest.importorskip("kneed")
        X_train, _, y_train, _, feature_names = regression_data
        model = LinearRegression()
        model.fit(X_train, y_train)
        names, _ = shap_select(model, X_train, feature_names, task='regression')

        from shap_selection import select_by_knee_detection
        result = select_by_knee_detection(
            LinearRegression, X_train, y_train, feature_names, names,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2,
        )
        assert 'knee_fraction' in result
        assert 'selected_fraction' in result
        assert 'selected_features' in result

    def test_selected_features_are_subset(self, regression_data):
        pytest.importorskip("kneed")
        X_train, _, y_train, _, feature_names = regression_data
        model = LinearRegression()
        model.fit(X_train, y_train)
        names, _ = shap_select(model, X_train, feature_names, task='regression')

        from shap_selection import select_by_knee_detection
        result = select_by_knee_detection(
            LinearRegression, X_train, y_train, feature_names, names,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2,
        )
        assert set(result['selected_features']).issubset(set(feature_names))
        assert len(result['selected_features']) >= 1

    def test_knee_equals_selected_fraction(self, regression_data):
        pytest.importorskip("kneed")
        X_train, _, y_train, _, feature_names = regression_data
        model = LinearRegression()
        model.fit(X_train, y_train)
        names, _ = shap_select(model, X_train, feature_names, task='regression')

        from shap_selection import select_by_knee_detection
        result = select_by_knee_detection(
            LinearRegression, X_train, y_train, feature_names, names,
            task='regression', steps=[0.2, 0.4, 0.6, 0.8, 1.0], cv=2,
        )
        assert result['knee_fraction'] == result['selected_fraction']


# ---------------------------------------------------------------------------
# Default cv
# ---------------------------------------------------------------------------

class TestDefaults:
    def test_default_cv_is_3(self, regression_data):
        """keep_absolute should use cv=3 by default."""
        from unittest.mock import patch
        X_train, _, y_train, _, feature_names = regression_data
        model = LinearRegression()
        model.fit(X_train, y_train)
        names, _ = shap_select(model, X_train, feature_names, task='regression')

        captured = {}

        original_cvs = __import__(
            'sklearn.model_selection', fromlist=['cross_val_score']
        ).cross_val_score

        def mock_cvs(estimator, X, y, cv=None, scoring=None, **kw):
            captured['cv'] = cv
            return original_cvs(estimator, X, y, cv=cv, scoring=scoring, **kw)

        with patch('shap_selection._core.cross_val_score', mock_cvs):
            keep_absolute(
                LinearRegression, X_train, y_train, feature_names, names,
                task='regression', steps=[1.0],
            )

        assert captured['cv'] == 3

