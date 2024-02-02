"""resale-prop-analysis REST API."""

import os
from contextlib import asynccontextmanager
from typing import Any

import joblib
import wandb
from fastapi import FastAPI, HTTPException
from loguru import logger
from src.feat_eng import prepare_features_for_inference
from src.train import (
    ALIASES,
    PROJECT_NAME,
    REGISTERED_MODEL_NAME,
)

# Global variable to hold the loaded model
model = None
ENV = os.getenv("ENV", "stg")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_dir = "models"
    model_filename = f"rf_w10_model_{ENV}.pkl"
    model_path = os.path.join(model_dir, model_filename)

    # Create the model directory if it does not exist
    os.makedirs(model_dir, exist_ok=True)

    # Check if the model file already exists locally
    if not os.path.exists(model_path):
        # Download the model from wandb if it's not already downloaded
        try:
            print("xxxxxxxxxxxxxxxxxxxxx", os.getenv("WANDB_API_KEY"))
            wandb.login(key=os.getenv("WANDB_API_KEY"))
            run = wandb.init(project=PROJECT_NAME, job_type="inference")
            logger.info(f"Loading model from {REGISTERED_MODEL_NAME}...")
            artifact = run.use_artifact(
                f"model-registry/{REGISTERED_MODEL_NAME}:{ALIASES[0]}", type="model"
            )
            artifact_dir = artifact.download()
            wandb.finish()

            downloaded_model_files = [f for f in os.listdir(artifact_dir) if f.endswith(".pkl")]
            if not downloaded_model_files:
                raise FileNotFoundError("No .pkl file found in the artifact directory")

            os.rename(os.path.join(artifact_dir, downloaded_model_files[0]), model_path)
            logger.info("Model downloaded and moved to models directory")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise HTTPException(status_code=500, detail="Model downloading failed")

    # Load the model from the local file
    try:
        model = joblib.load(model_path)
        yield
        logger.info("Model loaded successfully from local file")
    except Exception as e:
        logger.error(f"Failed to load model from local file: {e}")
        raise HTTPException(status_code=500, detail="Model loading from local file failed")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root() -> str:
    """Read root."""
    return "Hello world"


@app.post("/predict")
async def make_prediction(input_body: dict[str, Any]) -> dict[str, Any]:
    """Make prediction.
    Example of input_body:
    {
        "input_data": {
            "town": "SENGKANG",
            "flat_type": "4 ROOM",
            "storey_range": "04 TO 06",
            "floor_area_sqm": 93,
            "flat_model": "Model A",
            "remaining_lease": 95
            }
        }

    Args:
        input_body (Dict[str, Any]): Input data.

    Returns:
        prediction (Dict[str, Any]):

    Raises:
        HTTPException: Model not loaded.
        HTTPException: Error during prediction.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        df = prepare_features_for_inference(input_body["input_data"])

        prediction = model.predict(df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
