# Bài tập lớn FIT4013 – Đề tài 9: Dự đoán khách hàng rời bỏ dịch vụ viễn thông

Bài toán: phân loại nhị phân, dự đoán khách hàng có rời bỏ dịch vụ (`Churn = Yes`) hay không,
dựa trên thông tin nhân khẩu học, dịch vụ đã đăng ký, hợp đồng và thanh toán.

## Thành viên nhóm
| Họ tên | MSSV | Nhiệm vụ | Commit chính |
|---|---|---|---|
| [điền] | [điền] | [điền] | [điền] |

## Cấu trúc
```
churn/                  mã nguồn chính
  config.py             tham số tái lập (random_state=42), đường dẫn, tên cột
  data.py               nạp và làm sạch dữ liệu (không học tham số)
  features.py           tiền xử lý trong Pipeline; đặc trưng kỹ thuật Luồng B
  models.py             baseline, KNN, Naive Bayes, Cây quyết định và lưới tham số
  evaluate.py           chia train/test một lần, GridSearchCV trên train, đánh giá test
  error_analysis.py     phân tích lỗi theo nhóm và các ca bỏ sót
  run_all.py            chạy toàn bộ, xuất bảng/hình vào results/
tests/test_pipeline.py  kiểm thử: làm sạch, chống rò rỉ, tái lập
tools_make_synthetic.py sinh dữ liệu tổng hợp CHỈ để kiểm thử mã
report/                 báo cáo LaTeX (XeLaTeX), dựa trên template của học phần
results/                kết quả sinh tự động (bảng, hình, run_info.json)
AI_USAGE.md             khai báo sử dụng công cụ AI
```

## Cài đặt và chạy
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# 1. Đặt dữ liệu vào data/ (xem data/README.md)
python -m unittest discover -s tests -v                  # kiểm thử
python -m churn.run_all                                  # thí nghiệm đầy đủ -> results/
```
Sau khi chạy, `results/run_info.json` ghi rõ tên tệp dữ liệu và cờ `la_du_lieu_tong_hop`.
Chỉ dùng kết quả khi cờ này là `false`.

Tái lập: mọi phép chia dữ liệu, CV và mô hình có tính ngẫu nhiên dùng `random_state=42`.
Phép chia train/test (80/20, phân tầng theo nhãn) được thực hiện một lần và dùng chung cho mọi thí nghiệm.

## Quy trình chống rò rỉ dữ liệu
- Điền thiếu bằng trung vị, chuẩn hóa và one-hot được đặt trong `Pipeline`, chỉ fit trên tập train.
- Siêu tham số chọn bằng 5-fold CV **chỉ trên tập train** (tiêu chí F1, lớp Churn là lớp thiểu số).
- Mô hình được chọn theo F1 trung bình CV trên tập train, **không** chọn theo tập test.
- Đặc trưng kỹ thuật (Luồng B) chỉ dùng thông tin của từng khách hàng, không dùng nhãn (có kiểm thử).
- Cột `customerID` bị loại (định danh, không có giá trị dự đoán).

## Dữ liệu và giấy phép
Dữ liệu thuộc bộ Telco Customer Churn (nguồn ghi trong báo cáo). Không đưa tệp dữ liệu lên repository
nếu giấy phép không cho phép; `data/` đã được loại khỏi Git qua `.gitignore`.

## Kết quả chính (dữ liệu Telco, N = 7.043)
Chạy `python -m churn.run_all` (random_state=42). Xem `results/metrics.csv` và báo cáo.
Mô hình được chọn theo F1 CV trên tập train: **KNN, luồng B** (F1 CV 0,603; trên tập test: F1 0,598, Recall 0,570, Precision 0,629, ROC-AUC 0,832).
Chênh lệch giữa các cấu hình hàng đầu rất nhỏ; xem mục Hạn chế trong báo cáo.
