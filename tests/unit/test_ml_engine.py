import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor

from app.core.exceptions import TrainingError
from app.engine.ml_engine import MLEngine


@pytest.fixture
def synthetic_regression_data():
    """Generates a small synthetic linear regression dataset."""
    rng = np.random.RandomState(42)
    X = rng.normal(size=(100, 3))
    # y = 3*x0 - 2*x1 + 0.5*x2 + noise
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(scale=0.1, size=100)

    X_df = pd.DataFrame(X, columns=["feat_a", "feat_b", "feat_c"])
    y_series = pd.Series(y, name="target")

    X_train, X_test = X_df.iloc[:80], X_df.iloc[80:]
    y_train, y_test = y_series.iloc[:80], y_series.iloc[80:]

    return X_train, X_test, y_train, y_test


def test_ml_engine_trains_pipeline_and_produces_metrics(synthetic_regression_data):
    """Verify that MLEngine fits the pipeline and produces predictions and valid metrics."""
    X_train, X_test, y_train, y_test = synthetic_regression_data
    engine = MLEngine()

    preprocessor = StandardScaler()
    model = LinearRegression()

    result = engine.train_and_evaluate(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        model=model
    )

    # 1. Pipeline check
    assert "pipeline" in result
    pipeline = result["pipeline"]
    assert isinstance(pipeline, Pipeline)
    assert hasattr(pipeline, "predict")

    # 2. Prediction test
    preds = pipeline.predict(X_test)
    assert len(preds) == len(X_test)

    # 3. Metrics check
    assert "metrics" in result
    metrics = result["metrics"]
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics

    assert isinstance(metrics["mae"], float)
    assert isinstance(metrics["rmse"], float)
    assert isinstance(metrics["r2"], float)

    assert metrics["mae"] > 0
    assert metrics["rmse"] > 0
    # For this synthetic dataset, R2 should be high (> 0.95)
    assert 0.95 <= metrics["r2"] <= 1.0


@pytest.mark.parametrize("model_class", [LinearRegression, Ridge, RandomForestRegressor])
def test_ml_engine_supports_different_regression_models(synthetic_regression_data, model_class):
    """Verify that different sklearn regression models can be passed to MLEngine."""
    X_train, X_test, y_train, y_test = synthetic_regression_data
    engine = MLEngine()

    if model_class == RandomForestRegressor:
        model = model_class(n_estimators=10, random_state=42)
    else:
        model = model_class()

    result = engine.train_and_evaluate(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=StandardScaler(),
        model=model
    )

    assert isinstance(result["pipeline"], Pipeline)
    assert "mae" in result["metrics"]
    assert "rmse" in result["metrics"]
    assert "r2" in result["metrics"]
    assert result["metrics"]["r2"] > 0.5


def test_ml_engine_training_failure_raises_training_error(synthetic_regression_data):
    """Verify that training or evaluation failure properly raises TrainingError."""
    X_train, X_test, y_train, y_test = synthetic_regression_data
    engine = MLEngine()

    # Mismatched lengths should trigger an error during fitting
    mismatched_y = y_train.iloc[:10]

    with pytest.raises(TrainingError) as exc_info:
        engine.train_and_evaluate(
            X_train=X_train,
            X_test=X_test,
            y_train=mismatched_y,
            y_test=y_test,
            preprocessor=StandardScaler(),
            model=LinearRegression()
        )

    assert "Training failed" in str(exc_info.value)
