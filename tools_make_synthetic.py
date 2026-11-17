"""CHỈ DÙNG ĐỂ KIỂM THỬ PIPELINE. Sinh dữ liệu tổng hợp cùng cấu trúc cột với bộ Telco Customer Churn.
KHÔNG phải dữ liệu thật; không dùng số liệu từ file này trong báo cáo kết quả chính thức."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 3000
out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/SYNTHETIC_telco_like.csv")

tenure = rng.integers(0, 73, n)
contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
monthly = np.where(internet == "Fiber optic", rng.normal(90, 12, n),
                   np.where(internet == "DSL", rng.normal(58, 10, n), rng.normal(22, 5, n))).clip(18, 120)
total = (monthly * (tenure + rng.uniform(0, 1, n))).round(2)
yn = lambda p: np.where(rng.uniform(size=n) < p, "Yes", "No")  # noqa: E731
services = {
    "PhoneService": yn(0.9), "MultipleLines": yn(0.45), "OnlineSecurity": yn(0.3),
    "OnlineBackup": yn(0.35), "DeviceProtection": yn(0.35), "TechSupport": yn(0.3),
    "StreamingTV": yn(0.4), "StreamingMovies": yn(0.4),
}
for k in list(services):
    services[k] = np.where(internet == "No", "No internet service" if k != "PhoneService" else services[k], services[k])
logit = (-0.6 - 0.04 * tenure + 0.8 * (contract == "Month-to-month") + 0.6 * (internet == "Fiber optic")
         + 0.01 * (monthly - 65) - 0.25 * (services["TechSupport"] == "Yes"))
churn = np.where(rng.uniform(size=n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
df = pd.DataFrame({
    "customerID": [f"S{i:05d}" for i in range(n)],
    "gender": rng.choice(["Female", "Male"], n), "SeniorCitizen": rng.integers(0, 2, n),
    "Partner": yn(0.48), "Dependents": yn(0.3), "tenure": tenure,
    **services, "InternetService": internet, "Contract": contract,
    "PaperlessBilling": yn(0.6),
    "PaymentMethod": rng.choice(["Electronic check", "Mailed check", "Bank transfer (automatic)",
                                 "Credit card (automatic)"], n),
    "MonthlyCharges": monthly.round(2),
    "TotalCharges": np.where(tenure == 0, " ", total.astype(str)),
    "Churn": churn,
})
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)
print(f"Đã ghi {out} ({len(df)} dòng). ĐÂY LÀ DỮ LIỆU TỔNG HỢP.")
