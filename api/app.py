from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

# Create FastAPI instance
app = FastAPI()

# Load the trained model from disk
model = load_model("models/reentry_lstm.h5")

# Define input format
class SequenceInput(BaseModel):
    sequence: List[List[float]]  # 2D list = sequence of timesteps

@app.post("/predict")
def predict(input_data: SequenceInput):
    try:
        # Convert input to NumPy array
        seq = np.array(input_data.sequence)

        # Wrap in a batch + ragged tensor (1 sequence only)
        ragged_seq = tf.ragged.constant([seq])

        # Make prediction
        prob = model.predict(ragged_seq)[0][0]

        # Binary decision
        prediction = 1 if prob > 0.5 else 0

        return {
            "reentry_probability": float(prob),
            "prediction": bool(prediction)
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
