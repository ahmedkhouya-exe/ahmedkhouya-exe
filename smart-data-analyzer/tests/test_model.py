import pytest

from src.model import load_data, train_and_evaluate


def test_dataset_has_rows_and_features():
    X, y, feature_names, target_names = load_data()
    assert len(X) > 0
    assert X.shape[1] == len(feature_names)
    assert len(set(y)) == len(target_names)


def test_training_returns_valid_accuracy():
    result = train_and_evaluate(n_estimators=20, test_size=0.25)
    assert 0.0 <= result["accuracy"] <= 1.0
    assert len(result["predictions"]) == len(result["y_test"])


def test_invalid_tree_count_is_rejected():
    with pytest.raises(ValueError, match="n_estimators"):
        train_and_evaluate(n_estimators=0)


def test_invalid_test_size_is_rejected():
    with pytest.raises(ValueError, match="test_size"):
        train_and_evaluate(test_size=0.75)
