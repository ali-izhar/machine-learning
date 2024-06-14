from typing import List, Union, Tuple
import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Constants
RANDOM_STATE = 55
DATA_PATH = "dataset/heart.csv"


def load_data(
    file_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and preprocess the dataset."""
    try:
        df = pd.read_csv(file_path)
        cat_variables = [
            "Sex",
            "ChestPainType",
            "RestingECG",
            "ExerciseAngina",
            "ST_Slope",
        ]
        df = pd.get_dummies(data=df, prefix=cat_variables, columns=cat_variables)
        features = [x for x in df.columns if x != "HeartDisease"]
        X_train, X_val, y_train, y_val = train_test_split(
            df[features], df["HeartDisease"], train_size=0.8, random_state=RANDOM_STATE
        )
        return X_train, X_val, y_train, y_val
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def plot_metrics(
    parameter_list: List[Union[int, None]],
    accuracy_list_train: List[float],
    accuracy_list_val: List[float],
    param_name: str,
) -> None:
    """Plot train vs validation metrics."""
    plt.title("Train vs Validation metrics")
    plt.xlabel(param_name)
    plt.ylabel("accuracy")
    plt.xticks(ticks=range(len(parameter_list)), labels=parameter_list)
    plt.plot(accuracy_list_train)
    plt.plot(accuracy_list_val)
    plt.legend(["Train", "Validation"])
    plt.show()


def predict(model, input_data: pd.DataFrame) -> pd.Series:
    """Predict the output using the given model and input data."""
    try:
        return model.predict(input_data)
    except Exception as e:
        raise Exception(f"Error in prediction: {e}")


def hyperparameter_tuning(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    if model_name == "xgboost":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 4, 5],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
        }
        model = XGBClassifier(random_state=RANDOM_STATE)
    elif model_name == "random_forest":
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5],
            "min_samples_split": [2, 5, 10],
        }
        model = RandomForestClassifier(random_state=RANDOM_STATE)
    elif model_name == "decision_tree":
        param_grid = {"max_depth": [3, 4, 5], "min_samples_split": [2, 5, 10]}
        model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    else:
        raise ValueError("Model not supported")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=5,
        n_jobs=-1,
        verbose=2,
    )
    grid_search.fit(X_train, y_train)

    print(f"Best parameters found: {grid_search.best_params_}")
    return grid_search.best_estimator_


def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), feature_names[indices], rotation=90)
    plt.show()


def load_model(model_name: str, directory: str = "models"):
    """Load a saved model from a file."""
    try:
        model_path = os.path.join(directory, f"{model_name}.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} does not exist.")
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")
