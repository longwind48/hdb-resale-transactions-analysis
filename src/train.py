"""This script is used for training a RandomForestRegressor model to predict the HDB resale prices.
It uses data from the 1990 to 2023, performs data cleaning feature engineering and
model training steps. The script supports hyperparameters and metrics in wandb and saves
the best model.

Example usage:
poetry run python -m src.train --wand-config-path config/sweep_rf.yaml --log-level INFO
"""

import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import typer
import wandb
import yaml
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from src.feat_eng import clean_data, clean_label, prepare_features

app = typer.Typer()
PROJECT_NAME = "hdb-resale-price-prediction"
REGISTERED_MODEL_NAME = "rf-w10-model"
ALIASES = ["latest"]
best_model_performance = None
best_model = None
best_run_id = None  # Store the ID of the run with the best model


def within_10(model, X_test, y_test):
    """Calculate the percentage of predictions within 10% of the actual price."""
    preds = model.predict(X_test)
    err = np.abs((preds / y_test) - 1)
    w10 = err < 0.1
    return w10.mean().round(3)


@app.command()
def main(
    wand_config_path: str = typer.Option(
        ..., "--wand-config-path", help="Path to the wandb configuration file."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
):
    global best_model_performance, best_model

    logger.remove()
    logger.add(sys.stderr, level=log_level.upper())

    with open(wand_config_path) as file:
        sweep_config = yaml.load(file, Loader=yaml.FullLoader)
        print(sweep_config)
    sweep_id = wandb.sweep(sweep=sweep_config, project=PROJECT_NAME)
    wandb.agent(sweep_id, function=train_model, count=2)

    if best_model is not None:
        save_best_model()


def train_model():
    global best_model_performance, best_model, best_run_id

    run = wandb.init(project=PROJECT_NAME)

    # Donwloaded from src/download_resale_hdb_dataset.py cli script
    df = pd.read_parquet("data/raw/resale_hdb_data.parquet")

    # Downloaded from https://tablebuilder.singstat.gov.sg/table/TS/M212882
    df_cpi = (
        pd.read_csv("data/raw/cpi_housing.csv", index_col=0).iloc[9:757, :1].reset_index(drop=False)
    )
    df_cpi.columns = ["month", "cpi"]

    # Clean data
    df = clean_data(df)
    df = clean_label(df, df_cpi)

    # Add features
    X_train, X_test, y_train, y_test = prepare_features(df)

    # instead of defining hard values
    n_estimators = wandb.config.n_estimators
    max_depth = wandb.config.max_depth
    min_samples_split = wandb.config.min_samples_split
    min_samples_leaf = wandb.config.min_samples_leaf

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )

    logger.info("Training model...")
    model.fit(X_train, y_train)

    logger.info("Logging metrics...")
    w10 = within_10(model, X_test, y_test)

    run.log(
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "within_10": w10,
        }
    )

    if best_model_performance is None or w10 > best_model_performance:
        best_model_performance = w10
        best_model = model
        best_run_id = run.id

    run.finish()


def save_best_model():
    global best_model_performance, best_model, best_run_id

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"best_model_w10_{best_model_performance}_date_{timestamp}.pkl"
    model_path = "models/" + model_filename

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    run = wandb.init(project=PROJECT_NAME)
    run.link_model(path=model_path, registered_model_name=REGISTERED_MODEL_NAME, aliases=ALIASES)

    logger.info(f"Best model saved to wandb model registry as {REGISTERED_MODEL_NAME}.")


if __name__ == "__main__":
    app()
