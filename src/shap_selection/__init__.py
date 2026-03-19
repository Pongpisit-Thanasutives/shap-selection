"""
shap_selection
==============
Feature selection via SHAP values.

Based on:
  Marcilio-Jr & Eler, "From explanations to feature selection: assessing
  SHAP values as feature selection mechanism", SIBGRAPI 2020.
"""

from ._core import (
    shap_select,
    keep_absolute,
    select_by_keep_absolute,
    select_by_knee_detection,
    auto_select,
    compute_criterion,
    shap_threshold_select,
    apply_feature_selection,
    KNEE_METHODS,
    CRITERION_SCORERS,
    INSAMPLE_CRITERIA,
)

__version__ = "1.0.2"

__all__ = [
    "shap_select",
    "keep_absolute",
    "select_by_keep_absolute",
    "select_by_knee_detection",
    "auto_select",
    "compute_criterion",
    "shap_threshold_select",
    "apply_feature_selection",
    "KNEE_METHODS",
    "CRITERION_SCORERS",
    "INSAMPLE_CRITERIA",
]
