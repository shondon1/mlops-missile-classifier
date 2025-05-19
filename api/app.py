from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import tensorflow as tf
from keras.models import load_model
import numpy as np
import time
import logging
import os

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(title="Missile Reentry Classifier")

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# Define input data structure for a single track
class TrackData(BaseModel):
    sequence: List[List[float]]  # A 2D list: list of timesteps, each with features
    track_id: Optional[str] = "unknown"  # Optional track ID

# Define input for batch prediction
class BatchData(BaseModel):
    sequences: List[TrackData]  # A list of track data objects

# Load model when app starts
try:
    model_path = "models/reentry_lstm.h5"
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None  # We'll handle this in the endpoint

@app.get("/")
def home():
    """Health check endpoint"""
    return {"status": "online", "model": "reentry_classifier"}

@app.post("/predict")
def predict(data: TrackData):
    """
    Makes a prediction for a single missile track.
    
    Returns:
    - track_id: The ID of the track (if provided)
    - reentry_probability: Probability of being in reentry phase (0-1)
    - prediction: Boolean indicating if in reentry phase
    - inference_time_ms: Time taken to make prediction
    """
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Start timing for inference
        start_time = time.time()
        
        # Convert input to numpy array
        sequence = np.array(data.sequence)
        
        # Check if the sequence has the right shape
        if len(sequence.shape) != 2:
            raise HTTPException(status_code=400, detail="Sequence must be a 2D array")
        
        # Create a ragged tensor (for variable length sequences)
        ragged_tensor = tf.ragged.constant([sequence])
        
        # Make prediction 
        prediction = float(model.predict(ragged_tensor)[0][0])
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # in milliseconds
        
        # Log the prediction and time
        logger.info(f"Track {data.track_id}: Prediction {prediction:.4f}, Time: {inference_time:.2f}ms")
        
        # Return prediction and metadata
        return {
            "track_id": data.track_id,
            "reentry_probability": prediction,
            "prediction": bool(prediction > 0.5),  # True if reentry phase
            "inference_time_ms": inference_time
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict")
def batch_predict(data: BatchData):
    """
    Makes predictions for multiple missile tracks.
    
    Returns:
    - results: List of prediction results
    - batch_size: Number of sequences processed
    - total_inference_time_ms: Total time taken for all predictions
    - avg_inference_time_ms: Average time per prediction
    """
    try:
        # Check if model is loaded
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Start timing for batch inference
        start_time = time.time()
        
        results = []
        for track_data in data.sequences:
            # Convert to numpy array
            sequence = np.array(track_data.sequence)
            
            # Create a ragged tensor for this sequence
            ragged_tensor = tf.ragged.constant([sequence])
            
            # Make prediction
            pred_start_time = time.time()
            prediction = float(model.predict(ragged_tensor)[0][0])
            pred_time = (time.time() - pred_start_time) * 1000  # ms
            
            # Add to results
            results.append({
                "track_id": track_data.track_id,
                "reentry_probability": prediction,
                "prediction": bool(prediction > 0.5),
                "inference_time_ms": pred_time
            })
        
        # Calculate total inference time
        total_inference_time = (time.time() - start_time) * 1000  # ms
        avg_inference_time = total_inference_time / len(data.sequences) if data.sequences else 0
        
        # Log the batch prediction
        logger.info(f"Batch prediction: {len(results)} sequences, total time: {total_inference_time:.2f}ms")
        
        return {
            "results": results,
            "batch_size": len(results),
            "total_inference_time_ms": total_inference_time,
            "avg_inference_time_ms": avg_inference_time
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))