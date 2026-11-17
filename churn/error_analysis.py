"""Phân tích lỗi trên tập kiểm thử: tỷ lệ dự đoán sai theo nhóm và các ca sai điển hình."""
import pandas as pd

from .config import TARGET


def error_by_group(test_df: pd.DataFrame, y_pred, group_cols) -> pd.DataFrame:
    out = []
    df = test_df.copy()
    df["_pred"] = y_pred
    df["_err"] = (df["_pred"] != df[TARGET]).astype(int)
    df["_fn"] = ((df[TARGET] == 1) & (df["_pred"] == 0)).astype(int)
    df["_fp"] = ((df[TARGET] == 0) & (df["_pred"] == 1)).astype(int)
    for col in group_cols:
        g = df.groupby(col).agg(so_mau=("_err", "size"), ti_le_sai=("_err", "mean"),
                                so_FN=("_fn", "sum"), so_FP=("_fp", "sum"),
                                ti_le_churn_thuc=(TARGET, "mean")).reset_index()
        g.insert(0, "nhom", col)
        g = g.rename(columns={col: "gia_tri"})
        out.append(g)
    return pd.concat(out, ignore_index=True)


def false_negative_examples(test_df: pd.DataFrame, y_prob, n: int = 10) -> pd.DataFrame:
    """Các khách hàng thực sự rời bỏ nhưng mô hình có xác suất rời bỏ thấp nhất (bỏ sót nhiều nhất)."""
    df = test_df.copy()
    df["xac_suat_churn"] = y_prob
    fn = df[df[TARGET] == 1].sort_values("xac_suat_churn").head(n)
    cols = ["tenure", "Contract", "InternetService", "MonthlyCharges", "xac_suat_churn"]
    return fn[cols].reset_index(drop=True)
