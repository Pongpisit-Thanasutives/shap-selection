"""
shap_selection
==============
Feature selection via SHAP values, following:
  Marcilio-Jr & Eler, "From explanations to feature selection:
  assessing SHAP values as feature selection mechanism", SIBGRAPI 2020.
"""

from ._core import (
    shap_select,
    keep_absolute,
    select_by_keep_absolute,
    select_by_knee_detection,
    auto_select,
    shap_threshold_select,
    apply_feature_selection,
    KNEE_METHODS,
)

__version__ = "0.4.0"

__all__ = [
    "shap_select",
    "keep_absolute",
    "select_by_keep_absolute",
    "select_by_knee_detection",
    "auto_select",
    "shap_threshold_select",
    "apply_feature_selection",
    "KNEE_METHODS",
]
