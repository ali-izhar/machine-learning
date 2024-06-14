import joblib
import os
from typing import Tuple
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from utils import plot_metrics, load_data

# Constants
RANDOM_STATE = 55
MODEL_DIR = "models/"
DATA_PATH = "dataset/heart.csv"


def train_and_evaluate_model(
    model: object,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> Tuple[float, float]:
    """Train and evaluate the model."""
    try:
        model.fit(X_train, y_train)
        predictions_train = model.predict(X_train)
        predictions_val = model.predict(X_val)
        accuracy_train = accuracy_score(predictions_train, y_train)
        accuracy_val = accuracy_score(predictions_val, y_val)
        return accuracy_train, accuracy_val
    except Exception as e:
        raise Exception(f"Error training model: {e}")


def save_model(model, model_name: str, directory: str = "models") -> None:
    """Save the trained model to a file."""
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path = os.path.join(directory, f"{model_name}.pkl")
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
    except Exception as e:
        raise Exception(f"Error saving model: {e}")


def get_model_performance(model_name: str):
    """Get the performance of the specified model."""
    try:
        X_train, X_val, y_train, y_val = load_data(DATA_PATH)
        if model_name == "decision_tree":
            model = decision_tree_model(X_train, y_train, X_val, y_val)
        elif model_name == "random_forest":
            model = random_forest_model(X_train, y_train, X_val, y_val)
        elif model_name == "xgboost":
            model, _ = xgboost_model(X_train, y_train, X_val, y_val)
        else:
            raise ValueError("Model not supported")
        return model
    except Exception as e:
        raise Exception(f"Error in model performance: {e}")


def decision_tree_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
):
    """Train and evaluate a Decision Tree model."""
    try:
        min_samples_split_list = [2, 10, 30, 50, 100, 200, 300, 700]
        max_depth_list = [1, 2, 3, 4, 8, 16, 32, 64, None]

        accuracy_list_train = []
        accuracy_list_val = []
        for min_samples_split in min_samples_split_list:
            model = DecisionTreeClassifier(
                min_samples_split=min_samples_split, random_state=RANDOM_STATE
            )
            accuracy_train, accuracy_val = train_and_evaluate_model(
                model, X_train, y_train, X_val, y_val
            )
            accuracy_list_train.append(accuracy_train)
            accuracy_list_val.append(accuracy_val)

        plot_metrics(
            min_samples_split_list,
            accuracy_list_train,
            accuracy_list_val,
            "min_samples_split",
        )

        accuracy_list_train = []
        accuracy_list_val = []
        for max_depth in max_depth_list:
            model = DecisionTreeClassifier(
                max_depth=max_depth, random_state=RANDOM_STATE
            )
            accuracy_train, accuracy_val = train_and_evaluate_model(
                model, X_train, y_train, X_val, y_val
            )
            accuracy_list_train.append(accuracy_train)
            accuracy_list_val.append(accuracy_val)

        plot_metrics(
            max_depth_list, accuracy_list_train, accuracy_list_val, "max_depth"
        )

        best_model = DecisionTreeClassifier(
            min_samples_split=50, max_depth=4, random_state=RANDOM_STATE
        )
        best_model.fit(X_train, y_train)
        return best_model
    except Exception as e:
        raise Exception(f"Error training Decision Tree model: {e}")


def random_forest_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
):
    """Train and evaluate a Random Forest model."""
    try:
        min_samples_split_list = [2, 10, 30, 50, 100, 200, 300, 700]
        max_depth_list = [2, 4, 8, 16, 32, 64, None]
        n_estimators_list = [10, 50, 100, 500]

        accuracy_list_train = []
        accuracy_list_val = []
        for min_samples_split in min_samples_split_list:
            model = RandomForestClassifier(
                min_samples_split=min_samples_split, random_state=RANDOM_STATE
            )
            accuracy_train, accuracy_val = train_and_evaluate_model(
                model, X_train, y_train, X_val, y_val
            )
            accuracy_list_train.append(accuracy_train)
            accuracy_list_val.append(accuracy_val)

        plot_metrics(
            min_samples_split_list,
            accuracy_list_train,
            accuracy_list_val,
            "min_samples_split",
        )

        accuracy_list_train = []
        accuracy_list_val = []
        for max_depth in max_depth_list:
            model = RandomForestClassifier(
                max_depth=max_depth, random_state=RANDOM_STATE
            )
            accuracy_train, accuracy_val = train_and_evaluate_model(
                model, X_train, y_train, X_val, y_val
            )
            accuracy_list_train.append(accuracy_train)
            accuracy_list_val.append(accuracy_val)

        plot_metrics(
            max_depth_list, accuracy_list_train, accuracy_list_val, "max_depth"
        )

        accuracy_list_train = []
        accuracy_list_val = []
        for n_estimators in n_estimators_list:
            model = RandomForestClassifier(
                n_estimators=n_estimators, random_state=RANDOM_STATE
            )
            accuracy_train, accuracy_val = train_and_evaluate_model(
                model, X_train, y_train, X_val, y_val
            )
            accuracy_list_train.append(accuracy_train)
            accuracy_list_val.append(accuracy_val)

        plot_metrics(
            n_estimators_list, accuracy_list_train, accuracy_list_val, "n_estimators"
        )

        best_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_split=10,
            random_state=RANDOM_STATE,
        )
        best_model.fit(X_train, y_train)
        return best_model
    except Exception as e:
        raise Exception(f"Error training Random Forest model: {e}")


def xgboost_model(
    X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series
):
    """Train and evaluate an XGBoost model."""
    try:
        n = int(len(X_train) * 0.8)
        X_train_fit, X_train_eval, y_train_fit, y_train_eval = (
            X_train[:n],
            X_train[n:],
            y_train[:n],
            y_train[n:],
        )

        xgb_model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.01,
            verbosity=1,
            random_state=RANDOM_STATE,
            early_stopping_rounds=10,
        )
        xgb_model.fit(X_train_fit, y_train_fit, eval_set=[(X_train_eval, y_train_eval)])
        return xgb_model, xgb_model.best_iteration
    except Exception as e:
        raise Exception(f"Error training XGBoost model: {e}")
