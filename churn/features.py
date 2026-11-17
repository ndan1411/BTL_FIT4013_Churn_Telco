"""Tiền xử lý và biểu diễn đặc trưng.

Luồng A (cơ bản): đặc trưng gốc; số: điền trung vị + chuẩn hóa; phân loại: one-hot.
Luồng B (cải tiến): thêm đặc trưng có lý giải:
  - num_services: số dịch vụ đăng ký (giả thuyết: khách dùng nhiều dịch vụ gắn bó lâu hơn)
  - charge_per_tenure: TotalCharges / (tenure + 1), đo mức chi tiêu trung bình theo thời gian gắn bó
  - tenure_group: nhóm thời gian gắn bó (quan hệ không tuyến tính với nguy cơ rời bỏ)
  - is_month_to_month: hợp đồng theo tháng (nhóm có nguy cơ rời bỏ cao nhất trong EDA)
Tất cả đặc trưng mới chỉ dùng thông tin của chính khách hàng đó, không dùng nhãn: không rò rỉ.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import NUMERIC_COLS, SERVICE_COLS, TARGET


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    services_yes = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["num_services"] = services_yes
    df["charge_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["tenure_group"] = pd.cut(df["tenure"], bins=[-1, 12, 24, 48, np.inf],
                                labels=["0-12", "13-24", "25-48", "49+"]).astype(str)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def split_xy(df: pd.DataFrame):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_preprocessor(X: pd.DataFrame, scale_numeric: bool = True) -> ColumnTransformer:
    """Xây dựng bộ tiền xử lý từ cấu trúc cột của X. Được fit trên tập huấn luyện trong Pipeline."""
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


def make_pipeline(model, X: pd.DataFrame, scale_numeric: bool = True) -> Pipeline:
    return Pipeline([("prep", build_preprocessor(X, scale_numeric)), ("model", model)])
