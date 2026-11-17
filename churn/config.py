"""Cấu hình chung: tên cột, tham số tái lập và đường dẫn."""
from pathlib import Path

RANDOM_STATE = 42          # dùng cho mọi phép chia dữ liệu, CV và mô hình có tính ngẫu nhiên
TEST_SIZE = 0.2            # tỷ lệ tập kiểm thử (phân tầng theo nhãn)
CV_FOLDS = 5               # số fold cho GridSearchCV trên tập huấn luyện
SCORING = "f1"             # tiêu chí chọn siêu tham số (lớp Churn là lớp thiểu số)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

ID_COL = "customerID"
TARGET = "Churn"
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
SERVICE_COLS = [
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]
