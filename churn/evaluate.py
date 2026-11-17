"""Huấn luyện, tinh chỉnh siêu tham số (chỉ trên tập huấn luyện bằng CV) và đánh giá trên tập kiểm thử."""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .config import CV_FOLDS, RANDOM_STATE, SCORING, TARGET, TEST_SIZE
from .features import add_engineered, make_pipeline, split_xy
from .models import ALL_MODELS, baseline


def make_split(df: pd.DataFrame):
    """Chia train/test MỘT LẦN, phân tầng theo nhãn. Mọi thí nghiệm dùng cùng một cách chia."""
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET],
                                         random_state=RANDOM_STATE)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def feature_sets(train_df, test_df):
    """Luồng A: đặc trưng gốc. Luồng B: thêm đặc trưng kỹ thuật (tính trên từng dòng, không dùng nhãn)."""
    return {
        "A_co_ban": (train_df, test_df),
        "B_cai_tien": (add_engineered(train_df), add_engineered(test_df)),
    }


def metrics_of(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def run_experiments(train_df, test_df, cv_folds=CV_FOLDS):
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    rows, best_params, predictions = [], [], {}

    for fs_name, (tr, te) in feature_sets(train_df, test_df).items():
        X_tr, y_tr = split_xy(tr)
        X_te, y_te = split_xy(te)

        for model_name, factory in ALL_MODELS.items():
            model, grid = factory()
            pipe = make_pipeline(model, X_tr)
            search = GridSearchCV(pipe, grid, cv=cv, scoring=SCORING, n_jobs=-1, refit=True)
            search.fit(X_tr, y_tr)               # chỉ dùng tập huấn luyện
            est = search.best_estimator_
            y_pred = est.predict(X_te)
            y_prob = est.predict_proba(X_te)[:, 1]
            m = metrics_of(y_te, y_pred, y_prob)
            rows.append({"DacTrung": fs_name, "MoHinh": model_name, "CV_F1": search.best_score_, **m})
            best_params.append({"DacTrung": fs_name, "MoHinh": model_name,
                                **{k.replace("model__", ""): v for k, v in search.best_params_.items()}})
            predictions[(fs_name, model_name)] = (y_te.values, y_pred, y_prob, est)

        # Mốc so sánh (baseline): lớp đa số
        model, _ = baseline()
        pipe = make_pipeline(model, X_tr).fit(X_tr, y_tr)
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]
        m = metrics_of(y_te, y_pred, y_prob)
        rows.append({"DacTrung": fs_name, "MoHinh": "Baseline_LopDaSo", "CV_F1": np.nan, **m})
        predictions[(fs_name, "Baseline_LopDaSo")] = (y_te.values, y_pred, y_prob, pipe)

    results = pd.DataFrame(rows)
    return results, pd.DataFrame(best_params), predictions


def pick_best(results: pd.DataFrame) -> tuple:
    """Chọn cấu hình tốt nhất theo F1 trung bình trong CV trên TẬP HUẤN LUYỆN.
    Không chọn theo tập kiểm thử để tránh rò rỉ thông tin từ tập kiểm thử vào quyết định."""
    cand = results[results["MoHinh"] != "Baseline_LopDaSo"]
    r = cand.sort_values("CV_F1", ascending=False).iloc[0]
    return r["DacTrung"], r["MoHinh"]
