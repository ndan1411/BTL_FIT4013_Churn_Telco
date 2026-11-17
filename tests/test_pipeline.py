"""Kiểm thử đơn vị cho dữ liệu, tiền xử lý và tính tái lập. Chạy: python -m unittest discover -s tests -v"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from churn.config import NUMERIC_COLS, TARGET  # noqa: E402
from churn.data import clean  # noqa: E402
from churn.evaluate import make_split  # noqa: E402
from churn.features import add_engineered, build_preprocessor, split_xy  # noqa: E402


def tiny_raw(n=40, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "customerID": [f"C{i}" for i in range(n)],
        "gender": rng.choice(["Female", "Male"], n), "SeniorCitizen": rng.integers(0, 2, n),
        "Partner": rng.choice(["Yes", "No"], n), "Dependents": rng.choice(["Yes", "No"], n),
        "tenure": rng.integers(0, 70, n),
        "PhoneService": rng.choice(["Yes", "No"], n), "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n),
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(["Electronic check", "Mailed check"], n),
        "MonthlyCharges": rng.uniform(20, 110, n).round(2),
        "TotalCharges": [" " if t == 0 else f"{t * 50.0:.2f}" for t in rng.integers(0, 70, n)],
        "Churn": rng.choice(["Yes", "No"], n),
    })


class DataTest(unittest.TestCase):
    def test_clean_maps_target_and_drops_id(self):
        df = clean(tiny_raw())
        self.assertNotIn("customerID", df.columns)
        self.assertTrue(set(df[TARGET].unique()) <= {0, 1})
        self.assertTrue(pd.api.types.is_numeric_dtype(df["TotalCharges"]))

    def test_blank_total_charges_become_missing_not_zero(self):
        raw = tiny_raw(n=10)
        raw.loc[0, "TotalCharges"] = " "
        df = clean(raw)
        self.assertTrue(pd.isna(df["TotalCharges"].iloc[0]))

    def test_unexpected_label_raises(self):
        raw = tiny_raw(n=10)
        raw.loc[0, "Churn"] = "Maybe"
        with self.assertRaises(ValueError):
            clean(raw)


class LeakageTest(unittest.TestCase):
    def test_engineered_features_do_not_use_target(self):
        df = clean(tiny_raw(seed=1))
        flipped = df.copy()
        flipped[TARGET] = 1 - flipped[TARGET]
        a = add_engineered(df).drop(columns=[TARGET])
        b = add_engineered(flipped).drop(columns=[TARGET])
        pd.testing.assert_frame_equal(a, b)

    def test_imputer_and_scaler_learn_only_from_train(self):
        df = clean(tiny_raw(n=200, seed=2))
        train_df, test_df = make_split(df)
        X_tr, _ = split_xy(train_df)
        prep = build_preprocessor(X_tr, scale_numeric=True).fit(X_tr)
        num_tr = prep.named_transformers_["num"]
        scaler = num_tr.named_steps["scale"]
        imputer = num_tr.named_steps["impute"]
        numeric_cols = [c for c in X_tr.columns if pd.api.types.is_numeric_dtype(X_tr[c])]
        expected_median = X_tr[numeric_cols].median().values
        np.testing.assert_allclose(imputer.statistics_, expected_median)
        # trung bình của scaler phải bằng trung bình TRAIN (đã điền trung vị), không phải của toàn bộ dữ liệu
        filled_train = X_tr[numeric_cols].fillna(pd.Series(expected_median, index=numeric_cols))
        np.testing.assert_allclose(scaler.mean_, filled_train.mean().values, rtol=1e-8)


class ReproducibilityTest(unittest.TestCase):
    def test_split_is_reproducible_and_stratified(self):
        df = clean(tiny_raw(n=300, seed=3))
        tr1, te1 = make_split(df)
        tr2, te2 = make_split(df)
        pd.testing.assert_frame_equal(tr1, tr2)
        pd.testing.assert_frame_equal(te1, te2)
        self.assertAlmostEqual(tr1[TARGET].mean(), te1[TARGET].mean(), delta=0.05)


if __name__ == "__main__":
    unittest.main()
