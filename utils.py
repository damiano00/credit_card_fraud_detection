import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import joblib
from pathlib import Path
import shutil
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def __get_dataset():
    df = pd.read_csv("creditcard_2023.csv", index_col=False)
    return df.drop(["id", "Class"], axis=1), df["Class"]


X, y = __get_dataset()
RANDOM_STATE = 42
RESULT_PATH = "results"
RETRAIN = False


models = {
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(
        random_state=RANDOM_STATE, n_estimators=90, max_depth=18, n_jobs=-1
    ),
    "K Nearest Neighbors": KNeighborsClassifier(n_neighbors=2, n_jobs=-1),
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, n_jobs=-1),
}

pre_process = {
    "MinMax": MinMaxScaler(),
    "StdScaler": StandardScaler(),
    "PCA": PCA(n_components=12, random_state=RANDOM_STATE),
}


def get_splits(test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def evaluate_trained_model(model, X_eval, y_eval, save_dir, is_test, do_cross=True):
    y_pred = model.predict(X_eval)
    str_to_add = "test" if is_test else "train"
    report = f"classification {str_to_add} set\n"
    report += classification_report(y_eval, y_pred, digits=4)
    print(report)
    save_results(save_dir, report, is_test)
    cm = confusion_matrix(y_eval, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot()
    plt.savefig(Path(f"{save_dir}/{str_to_add}_confusion_matrix.png"), dpi=300)
    plt.show()
    if do_cross:
        # sturges = int(1 + np.log(len(X)))
        sturges = 10
        scores = cross_val_score(model, X, y, cv=sturges)
        cross_report = f"Cross evaluation: {scores}\n"
        cross_report += f"{scores.mean()} accuracy with a standard deviation of {scores.std()} in {sturges} folds"
        print(cross_report)
        report += cross_report
    save_results(save_dir, report, is_test)


# def evaluate_all_model(model, X_train, y_train, X_test, y_test, save_dir):
#     evaluate_trained_model(
#         model, X_train, y_train, save_dir, is_test=False, do_cross=False
#     )
#     evaluate_trained_model(model, X_test, y_test, save_dir, is_test=True, do_cross=True)


def evaluate_all_model(model, save_dir):
    X_train, X_test, y_train, y_test = get_splits()
    evaluate_trained_model(
        model, X_train, y_train, save_dir, is_test=False, do_cross=False
    )
    evaluate_trained_model(model, X_test, y_test, save_dir, is_test=True, do_cross=True)


def save_results(dir, report, isTest):
    dir = Path(dir)
    if not dir.exists():
        dir.mkdir(parents=True)
    str_to_add = "test" if isTest else "train"
    with open(f"{dir}/{str_to_add}_report.txt", "w") as f:
        f.write(report)


def create_dir(dir):
    dir = Path(dir)
    if dir.exists():
        shutil.rmtree(dir)
    dir.mkdir(parents=True)


def get_model(dir):
    dir = f"{RESULT_PATH}/{dir}"
    dir = Path(dir)
    if not dir.exists():
        return None
    pickle_file = [
        file
        for file in os.listdir(dir)
        if file.endswith(".pkl") or file.endswith(".joblib")
    ][0]
    dir = os.path.join(dir, pickle_file)
    with open(dir, "rb") as file:
        pipeline = joblib.load(file)
    return pipeline


def train_evaluate_model(model, X_train, y_train, X_test, y_test):
    model_str = str([v for _, v in model.steps]).replace("[", "").replace("]", "")
    dir = f"{RESULT_PATH}/{model_str}"
    model.fit(X_train, y_train)
    evaluate_trained_model(model, X_test, y_test, dir)
    return model


def evaluate_trained_model_custom_dir(model_dir, X_train, y_train, X_test, y_test):
    model = get_model(model_dir)
    model_dir = f"{RESULT_PATH}/{model_dir}"
    evaluate_all_model(model, X_train, y_train, X_test, y_test, model_dir)


def train_evaluate_model_custom_dir(model, X_train, y_train, X_test, y_test, dir):
    dir = f"{RESULT_PATH}/{dir}"
    model.fit(X_train, y_train)
    evaluate_all_model(model, X_train, y_train, X_test, y_test, dir)
    return model


def find_best_model(model, param_grid, search_method, cv=2):
    if search_method == "grid":
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=4,
        )
    else:
        grid_search = GridSearchCV(
            model,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1,
            verbose=4,
        )
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation accuracy: ", grid_search.best_score_)
    return best_model
