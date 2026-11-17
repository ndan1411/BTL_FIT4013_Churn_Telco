"""Các thuật toán đã học và lưới siêu tham số. Lưới nhỏ, lý giải được; không tìm kiếm rộng."""
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from .config import RANDOM_STATE


def baseline():
    """Mốc so sánh: luôn dự đoán lớp đa số (Không rời bỏ)."""
    return DummyClassifier(strategy="most_frequent"), {}


def knn():
    grid = {
        "model__n_neighbors": [5, 11, 21, 41],
        "model__weights": ["uniform", "distance"],
    }
    return KNeighborsClassifier(), grid


def naive_bayes():
    grid = {"model__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]}
    return GaussianNB(), grid


def decision_tree():
    grid = {
        "model__max_depth": [3, 5, 7, 10, None],
        "model__min_samples_leaf": [1, 20, 50],
    }
    return DecisionTreeClassifier(random_state=RANDOM_STATE), grid


ALL_MODELS = {"KNN": knn, "NaiveBayes": naive_bayes, "DecisionTree": decision_tree}
