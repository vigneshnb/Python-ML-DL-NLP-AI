"""Feature schema consistency and preprocessing maths."""

import json
from pathlib import Path

import numpy as np
import pytest

from app.features import FEATURE_NAMES, FEATURES, GROUPS
from app.services.preprocessing import PreprocessingError, Preprocessor

ROOT = Path(__file__).resolve().parent.parent
PRE_PATH = ROOT / "models" / "preprocessing.json"
PRE = json.loads(PRE_PATH.read_text(encoding="utf-8"))


def test_feature_order_matches_training_columns():
    assert list(FEATURE_NAMES) == [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
    ]
    assert PRE["feature_order"] == list(FEATURE_NAMES)


def test_every_feature_has_complete_definition():
    group_ids = {g["id"] for g in GROUPS}
    for f in FEATURES:
        assert f.label and f.description and f.validation
        assert f.dtype in {"float", "int", "binary"}
        assert f.group in group_ids
        assert f.min < f.max and f.step > 0
        assert f.min <= f.default <= f.max


def test_defaults_are_training_medians():
    for f, median in zip(FEATURES, PRE["train_median"]):
        assert f.default == pytest.approx(median, rel=1e-6, abs=1e-9), f.name


def test_training_range_lies_inside_hard_bounds():
    for f, lo, hi in zip(FEATURES, PRE["train_min"], PRE["train_max"]):
        assert f.min <= lo and hi <= f.max, f.name


def test_preprocessing_matches_model_file():
    import hashlib

    digest = hashlib.sha256((ROOT / "models" / PRE["model_file"]).read_bytes()).hexdigest()
    assert digest == PRE["model_sha256"], "model file differs from the one preprocessing.json was built for"


def test_transform_is_standard_scaling(sample):
    pre = Preprocessor(PRE_PATH)
    x = pre.transform(sample)
    raw = np.array([[sample[n] for n in FEATURE_NAMES]], dtype=np.float64)
    expected = (raw - np.array(PRE["mean"])) / np.array(PRE["scale"])
    assert x.shape == (1, 13) and x.dtype == np.float32
    np.testing.assert_allclose(x, expected, rtol=1e-6)


def test_transform_uses_schema_order_not_dict_order(sample):
    pre = Preprocessor(PRE_PATH)
    shuffled = dict(reversed(list(sample.items())))
    np.testing.assert_array_equal(pre.transform(sample), pre.transform(shuffled))


def test_transform_rejects_missing(sample):
    pre = Preprocessor(PRE_PATH)
    sample.pop("RM")
    with pytest.raises(ValueError):
        pre.transform(sample)


def test_out_of_training_range_detection(sample):
    pre = Preprocessor(PRE_PATH)
    assert pre.out_of_training_range(sample) == []
    sample["RM"] = 9.5
    assert pre.out_of_training_range(sample) == ["RM"]


def test_bad_preprocessing_files(tmp_path):
    with pytest.raises(PreprocessingError):
        Preprocessor(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(PreprocessingError):
        Preprocessor(bad)
    reordered = dict(PRE, feature_order=list(reversed(PRE["feature_order"])))
    p = tmp_path / "reordered.json"
    p.write_text(json.dumps(reordered), encoding="utf-8")
    with pytest.raises(PreprocessingError, match="order"):
        Preprocessor(p)
    zero = dict(PRE, scale=[0.0] * 13)
    p.write_text(json.dumps(zero), encoding="utf-8")
    with pytest.raises(PreprocessingError):
        Preprocessor(p)


def test_scaler_matches_sklearn_on_training_split():
    """Recompute the scaler from the cached Keras dataset (skipped if unavailable)."""
    sk = pytest.importorskip("sklearn.preprocessing")
    cache = Path.home() / ".keras" / "datasets" / "boston_housing.npz"
    if not cache.exists():
        pytest.skip("boston_housing.npz not cached")
    from keras.datasets import boston_housing

    (x_train, _), _ = boston_housing.load_data()
    scaler = sk.StandardScaler().fit(x_train)
    np.testing.assert_allclose(scaler.mean_, PRE["mean"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(scaler.scale_, PRE["scale"], rtol=0, atol=1e-10)
