"""
NBA Player Career Duration Prediction API

This API predicts the career duration of NBA players using a pre-trained XGBoost model.
It accepts a JSON list of player statistics and returns a JSON list of predictions.
Each prediction includes the index of the player, the predicted probability, and a message.

Endpoints:
1. `/`: Root endpoint providing a simple message.
2. `/predict/`: Endpoint for making predictions based on input player data.
"""

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import pandas as pd
import uvicorn
import logging
from typing import List
from pathlib import Path

from model import load_model, preprocess_data

# Constants
CLF_NAME = Path(__file__).parent / "models" / "xgboost.pkl"
SCALER_NAME = Path(__file__).parent / "models" / "robust_scaler.pkl"
HOST = "127.0.0.1"
PORT = 8000

# Mapping columns name between PlayerData and corresponding DataFrame columns expected by the preprocess_data function
COLUMNS_MAPPING = {
    "GP": "GP", "MIN": "MIN", "PTS": "PTS", "FGM": "FGM", "FGA": "FGA",
    "FGP": "FG%", "THREE_P_M": "3P Made", "THREE_P_A": "3PA", "THREE_P_P": "3P%",
    "FTM": "FTM", "FTA": "FTA", "FTP": "FT%", "OREB": "OREB", "DREB": "DREB",
    "REB": "REB", "AST": "AST", "STL": "STL", "BLK": "BLK", "TOV": "TOV",
}

# Load pre-trained model and scaler
CLF = load_model(filename=CLF_NAME)
SCALER = load_model(filename=SCALER_NAME)

# App instance
app = FastAPI()
logger = logging.getLogger("uvicorn")

class PlayerData(BaseModel):
    """
    Pydantic model representing input player data for prediction.

    Attributes:
    - GP: Games played.
    - MIN: Minutes played.
    - PTS: Points per game.
    - FGM: Field goals made.
    - FGA: Field goals attempted.
    - FGP: Field goal percentage.
    - THREE_P_M: Three-point shots made.
    - THREE_P_A: Three-point shots attempted.
    - THREE_P_P: Three-point shot percentage.
    - FTM: Free throws made.
    - FTA: Free throws attempted.
    - FTP: Free throw percentage.
    - OREB: Offensive rebounds.
    - DREB: Defensive rebounds.
    - REB: Total rebounds.
    - AST: Assists.
    - STL: Steals.
    - BLK: Blocks.
    - TOV: Turnovers.
    """
    GP: float
    MIN: float
    PTS: float
    FGM: float
    FGA: float
    FGP: float
    THREE_P_M: float
    THREE_P_A: float
    THREE_P_P: float
    FTM: float
    FTA: float
    FTP: float
    OREB: float
    DREB: float
    REB: float
    AST: float
    STL: float
    BLK: float
    TOV: float

    class Config:
        """
        Inputs of the prediction model.
        """
        json_schema_extra = {
            "example": {
                "GP": 38, "MIN": 7.5, "PTS": 1.9, "FGM": 0.7, "FGA": 2.3,
                "FGP": 31.8, "THREE_P_M": 0.1, "THREE_P_A": 0.6, "THREE_P_P": 20.8,
                "FTM": 0.3, "FTA": 0.7, "FTP": 44.4, "OREB": 0.4, "DREB": 0.5,
                "REB": 0.9, "AST": 0.4, "STL": 0.4, "BLK": 0.3, "TOV": 0.5,
            }
        }
        populate_by_name = True
        extra = "ignore"

@app.get("/", tags=["Root"], summary="Get a welcome message")
async def read_root():
    """
    Root endpoint returning a welcome message.
    """
    return {"message": "NBA Player Career Duration Prediction API"}

class Prediction(BaseModel):
    """
    A model representing a single prediction for an NBA player.

    Attributes:
    - index: The index of the player in the input list.
    - prediction: The predicted probability of the player staying in the NBA for the next 5 years. A value of 0 indicates that the player will not stay, while a value of 1 indicates that the player will stay.
    - msg: A message indicating whether the player will stay in the NBA for the next 5 years, based on the prediction value.
    """
    index: int
    prediction: float
    msg: str

    class Config:
        """
        Output of the prediction model.
        """
        json_schema_extra = {
            "example": {
                "index": 0,
                "prediction": 0.0,
                "msg": "Player will not stay in NBA for the next 5 years"
            }
        }

@app.post("/predict/", response_model=List[Prediction], tags=["Prediction"], summary="Predict NBA player career duration", response_description="A list of predictions for the provided player data")
async def predict(data_list: List[PlayerData]):
    """
    Endpoint for making predictions based on input player data.

    Parameters:
    - data_list: A JSON-like list of PlayerData objects containing player statistics.

    Returns:
    - A JSON-like list where each dictionary contains the index, the predicted probability, and a message.
    """
    try:
        predictions = [] 
        for i, data in enumerate(data_list):
            data_dict = jsonable_encoder(data)
            input_df = pd.DataFrame([data_dict]).rename(columns=COLUMNS_MAPPING)
            preprocessed_df = preprocess_data(input_df)
            scaled_df = SCALER.transform(preprocessed_df)
            prediction = CLF.predict(scaled_df)
            msg = 'Player will not stay in NBA for the next 5 years' if prediction[0] == 0 else 'Player will stay in NBA for the next 5 years'
            predictions.append({"index": i, "prediction": float(prediction[0]), "msg": msg})
        return predictions
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction"
        )

if __name__ == "__main__":
    # Run application using uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=HOST, port=PORT)
