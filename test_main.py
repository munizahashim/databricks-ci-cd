"""
Unit tests for the wine-quality-ml MLOps pipeline.

Covers the training, validation, and deployment (Challenger vs Champion) steps.
The validation and promotion tests import the real decision logic from
src/model_checks.py, so a change to that logic is caught here.
All tests run without a Databricks workspace, so CI completes in seconds.

Run from the bundle root:  pytest
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature

from model_checks import ACCURACY_THRESHOLD, should_promote, validation_failures


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def wine_features():
    """Small synthetic DataFrame matching the sklearn wine dataset schema."""
    rng = np.random.default_rng(42)
    n = 30
    return pd.DataFrame(
        {
            "alcohol": rng.uniform(11.0, 14.8, n),
            "malic_acid": rng.uniform(0.7, 5.8, n),
            "ash": rng.uniform(1.4, 3.2, n),
            "alcalinity_of_ash": rng.uniform(10.6, 30.0, n),
            "magnesium": rng.uniform(70.0, 162.0, n),
            "total_phenols": rng.uniform(0.9, 3.9, n),
            "flavanoids": rng.uniform(0.3, 5.1, n),
            "nonflavanoid_phenols": rng.uniform(0.1, 0.7, n),
            "proanthocyanins": rng.uniform(0.4, 3.6, n),
            "color_intensity": rng.uniform(1.3, 13.0, n),
            "hue": rng.uniform(0.5, 1.7, n),
            "od280/od315_of_diluted_wines": rng.uniform(1.3, 4.0, n),
            "proline": rng.uniform(278.0, 1680.0, n),
        }
    )


# ─── Training ────────────────────────────────────────────────────────────────


class TestTraining:
    """Training step: model trains on wine features and has an inferable signature."""

    def test_model_trains_and_produces_predictions(self, wine_features):
        rng = np.random.default_rng(42)
        labels = pd.Series(rng.integers(0, 3, len(wine_features)), name="label")

        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(wine_features, labels)
        predictions = model.predict(wine_features)

        assert len(predictions) == len(wine_features)
        assert set(predictions).issubset({0, 1, 2})

    def test_model_signature_is_inferred(self, wine_features):
        labels = pd.Series(np.random.default_rng(1).integers(0, 3, len(wine_features)))
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(wine_features, labels)
        predictions = model.predict(wine_features)

        signature = infer_signature(wine_features, predictions)

        assert signature is not None
        assert signature.inputs is not None
        assert signature.outputs is not None


# ─── Validation ──────────────────────────────────────────────────────────────


class TestValidationChecks:
    """Validation step: a version earns @Challenger only if every check passes."""

    def test_passes_with_input_example_and_required_tag(self):
        assert validation_failures(has_input_example=True, tags={"dataset": "sklearn_wine"}) == []

    def test_fails_when_input_example_missing(self):
        failures = validation_failures(has_input_example=False, tags={"dataset": "sklearn_wine"})
        assert failures == ["Model has no input example"]

    def test_fails_when_required_tag_missing(self):
        failures = validation_failures(has_input_example=True, tags={})
        assert failures == ["Required tag 'dataset' is missing from the model version"]

    def test_reports_every_failure_not_just_the_first(self):
        failures = validation_failures(has_input_example=False, tags={})
        assert len(failures) == 2

    def test_custom_required_tags_are_checked(self):
        failures = validation_failures(
            has_input_example=True,
            tags={"dataset": "sklearn_wine"},
            required_tags=("dataset", "owner"),
        )
        assert failures == ["Required tag 'owner' is missing from the model version"]


# ─── Deployment ──────────────────────────────────────────────────────────────


class TestPromotionDecision:
    """Deployment step: @Challenger replaces @Champion only if it performs at least as well."""

    def test_default_threshold_is_085(self):
        assert ACCURACY_THRESHOLD == 0.85

    # First deployment: no Champion yet, so the Challenger is compared to the threshold

    def test_first_model_promoted_above_threshold(self):
        assert should_promote(challenger_accuracy=0.91, champion_accuracy=None)

    def test_first_model_promoted_at_exact_threshold(self):
        assert should_promote(challenger_accuracy=0.85, champion_accuracy=None)

    def test_first_model_rejected_below_threshold(self):
        assert not should_promote(challenger_accuracy=0.8499, champion_accuracy=None)

    # Champion exists: the Challenger is compared to the Champion

    def test_challenger_promoted_when_better_than_champion(self):
        assert should_promote(challenger_accuracy=0.93, champion_accuracy=0.91)

    def test_challenger_promoted_when_equal_to_champion(self):
        assert should_promote(challenger_accuracy=0.91, champion_accuracy=0.91)

    def test_challenger_rejected_when_worse_than_champion(self):
        assert not should_promote(challenger_accuracy=0.89, champion_accuracy=0.91)
