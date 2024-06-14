import click
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utils import load_data, predict, hyperparameter_tuning, plot_feature_importance
from model import save_model


@click.group()
def cli():
    pass


@click.command()
@click.argument("model_name")
@click.option("--scale/--no-scale", default=True, help="Scale numerical features")
@click.option("--cv", default=5, help="Number of cross-validation folds")
def train(model_name: str, scale: bool, cv: int) -> None:
    """Train the specified model."""
    try:
        X_train, X_val, y_train, y_val = load_data("dataset/heart.csv")

        # Scale features
        if scale:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        # Hyperparameter tuning
        model = hyperparameter_tuning(model_name, X_train, y_train)

        # Save the best model
        save_model(model, model_name)
        print(f"{model_name} model trained and saved successfully!")

        # Testing the model with validation data
        predictions = predict(model, X_val)
        accuracy_val = (predictions == y_val).mean()
        print(f"Validation accuracy: {accuracy_val:.4f}")

        # Plot feature importance for XGBoost and RandomForest
        if model_name in ["xgboost", "random_forest"]:
            X_train_df = pd.DataFrame(
                X_train,
                columns=[col for col in load_data("dataset/heart.csv")[0].columns],
            )
            plot_feature_importance(model, X_train_df.columns)

    except Exception as e:
        print(f"Error training model: {e}")


cli.add_command(train)

if __name__ == "__main__":
    cli()

# python train.py train xgboost --scale --cv 5
