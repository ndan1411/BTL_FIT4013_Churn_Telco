"""Nạp và làm sạch dữ liệu. Chỉ các bước KHÔNG học tham số được thực hiện ở đây
(bỏ cột định danh, ép kiểu, mã hóa nhãn). Các bước học tham số (điền thiếu bằng trung vị,
chuẩn hóa, one-hot) nằm trong pipeline và chỉ được fit trên tập huấn luyện."""
import pandas as pd

from .config import DATA_PATH, ID_COL, TARGET


def load_raw(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # TotalCharges trong bộ gốc là chuỗi, có 11 ô khoảng trắng (khách có tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", None),
                                       errors="coerce")
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    if ID_COL in df.columns:
        df = df.drop(columns=[ID_COL])
    if df.duplicated().any():
        df = df.drop_duplicates()
    if set(df[TARGET].unique()) - {"Yes", "No"}:
        raise ValueError(f"Nhãn {TARGET} có giá trị ngoài Yes/No")
    df[TARGET] = (df[TARGET] == "Yes").astype(int)
    return df


def load_clean(path=DATA_PATH) -> pd.DataFrame:
    return clean(load_raw(path))


def describe_quality(df_raw: pd.DataFrame) -> dict:
    """Thống kê chất lượng dữ liệu cho phần EDA của báo cáo."""
    total_blank = (df_raw["TotalCharges"].astype(str).str.strip() == "").sum()
    return {
        "so_dong": int(df_raw.shape[0]),
        "so_cot": int(df_raw.shape[1]),
        "dong_trung_lap": int(df_raw.duplicated().sum()),
        "o_thieu_TotalCharges_dang_chuoi_trong": int(total_blank),
        "ti_le_churn": float((df_raw[TARGET] == "Yes").mean()),
    }
