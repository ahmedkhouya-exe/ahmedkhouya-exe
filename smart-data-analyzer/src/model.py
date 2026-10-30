"""Training utilities for the Smart Data Analyzer demo."""
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def load_data():
    """Return the Wine dataset as features, target, and feature names."""
    dataset = load_wine(as_frame=True)
    return (
        dataset.data,
        dataset.target,
        list(dataset.feature_names),
        list(dataset.target_names),
    )


def train_and_evaluate(
    n_estimators: int = 100,
    test_size: float = 0.25,
    random_state: int = 42,
):
    """Train a classifier and return evaluation outputs on held-out data."""
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")
    if not 0.1 <= test_size <= 0.5:
        raise ValueError("test_size must be between 0.1 and 0.5")

    X, y, feature_names, target_names = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return {
        "model": model,
        "accuracy": accuracy_score(y_test, predictions),
        "report": classification_report(
            y_test,
            predictions,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        ),
        "feature_names": feature_names,
        "target_names": target_names,
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions,
    }
