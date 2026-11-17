"""Chạy toàn bộ thí nghiệm và xuất kết quả cho báo cáo.

Cách chạy (từ thư mục gốc dự án):
    python -m churn.run_all                       # dùng data/WA_Fn-UseC_-Telco-Customer-Churn.csv
    python -m churn.run_all --data <đường dẫn csv>
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay  # noqa: E402

from .config import DATA_PATH, FIGURES_DIR, RESULTS_DIR, TARGET  # noqa: E402
from .data import clean, describe_quality, load_raw  # noqa: E402
from .error_analysis import error_by_group, false_negative_examples  # noqa: E402
from .evaluate import make_split, pick_best, run_experiments  # noqa: E402
from .features import add_engineered  # noqa: E402


def eda(df_raw: pd.DataFrame, df: pd.DataFrame):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    df_raw[TARGET].value_counts().plot.bar(ax=ax, color=["#4C72B0", "#DD8452"])
    ax.set_title("Phân phối nhãn Churn")
    ax.set_xlabel("Churn")
    ax.set_ylabel("Số khách hàng")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_phan_phoi_nhan.png", dpi=150)
    plt.close(fig)

    ct = df.groupby("Contract")[TARGET].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ct.plot.bar(ax=ax, color="#4C72B0")
    ax.set_title("Tỷ lệ rời bỏ theo loại hợp đồng (%)")
    ax.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_churn_theo_hop_dong.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    for lbl, color in [(0, "#4C72B0"), (1, "#DD8452")]:
        ax.hist(df.loc[df[TARGET] == lbl, "tenure"], bins=30, alpha=0.6, color=color,
                label="Churn=Yes" if lbl else "Churn=No")
    ax.set_title("Phân phối thời gian gắn bó (tenure) theo nhãn")
    ax.set_xlabel("tenure (tháng)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "eda_tenure.png", dpi=150)
    plt.close(fig)


def figures_for_best(predictions, best_fs, best_model, results):
    y_true, y_pred, y_prob, est = predictions[(best_fs, best_model)]
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["Không", "Rời bỏ"],
                                            cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Ma trận nhầm lẫn: {best_fs} / {best_model}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cm_best.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for (fs, mdl), (yt, _, yp, _) in predictions.items():
        if mdl == "Baseline_LopDaSo":
            continue
        RocCurveDisplay.from_predictions(yt, yp, name=f"{fs}/{mdl}", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("Đường ROC trên tập kiểm thử")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_cac_mo_hinh.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for (fs, mdl), (yt, _, yp, _) in predictions.items():
        if mdl == "Baseline_LopDaSo":
            continue
        PrecisionRecallDisplay.from_predictions(yt, yp, name=f"{fs}/{mdl}", ax=ax)
    ax.set_title("Đường Precision–Recall trên tập kiểm thử")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_cac_mo_hinh.png", dpi=150)
    plt.close(fig)

    # Độ quan trọng đặc trưng của cây quyết định luồng B (luôn vẽ để báo cáo có căn cứ)
    _, _, _, dt_est = predictions[("B_cai_tien", "DecisionTree")]
    names = dt_est.named_steps["prep"].get_feature_names_out()
    imp = dt_est.named_steps["model"].feature_importances_
    imp_s = pd.Series(imp, index=names).sort_values(ascending=False)
    imp_s.head(15).to_csv(RESULTS_DIR / "dt_feature_importance_top15.csv", header=["importance"])
    top = imp_s.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    top.plot.barh(ax=ax, color="#55A868")
    ax.set_title("10 đặc trưng quan trọng nhất (cây quyết định, luồng B)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "dt_feature_importance.png", dpi=150)
    plt.close(fig)


def write_tables(results, best_params):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    best_params.to_csv(RESULTS_DIR / "best_params.csv", index=False)
    cols = ["DacTrung", "MoHinh", "CV_F1", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "PR_AUC"]
    t = results[cols].copy()
    t.columns = ["Đặc trưng", "Mô hình", "F1 (CV)", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    tex = t.to_latex(index=False, float_format="%.3f", na_rep="--", escape=True)
    (RESULTS_DIR / "bang_ket_qua.tex").write_text(tex, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA_PATH))
    args = ap.parse_args()
    data_path = Path(args.data)
    is_synthetic = "SYNTHETIC" in data_path.name.upper()

    df_raw = load_raw(data_path)
    quality = describe_quality(df_raw)
    df = clean(df_raw)
    df_fe = add_engineered(df)
    eda(df_raw, df_fe)

    train_df, test_df = make_split(df)
    results, best_params, predictions = run_experiments(train_df, test_df)
    best_fs, best_model = pick_best(results)
    write_tables(results, best_params)
    figures_for_best(predictions, best_fs, best_model, results)

    y_true, y_pred, y_prob, _ = predictions[(best_fs, best_model)]
    test_fe = add_engineered(test_df)
    err = error_by_group(test_fe, y_pred, ["Contract", "InternetService", "tenure_group"])
    err.to_csv(RESULTS_DIR / "error_by_group.csv", index=False)
    false_negative_examples(test_fe, y_prob).to_csv(RESULTS_DIR / "false_negative_examples.csv", index=False)

    info = {
        "du_lieu": str(data_path.name),
        "la_du_lieu_tong_hop": is_synthetic,
        "chat_luong": quality,
        "so_dong_train": int(len(train_df)), "so_dong_test": int(len(test_df)),
        "mo_hinh_tot_nhat_theo_CV": {"dac_trung": best_fs, "mo_hinh": best_model},
        "random_state": 42, "test_size": 0.2, "cv_folds": 5, "scoring": "f1",
    }
    (RESULTS_DIR / "run_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(results[["DacTrung", "MoHinh", "CV_F1", "F1", "Recall", "Precision", "ROC_AUC", "PR_AUC"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
