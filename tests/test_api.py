"""
This is a test suite for validating the core functionality of the FastAPI application.

It includes:
- Health check endpoint testing
- Prediction endpoint testing (with mock model)
- Input validation error testing
- Batch prediction endpoint testing

"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
from api.app import app  # Import the FastAPI app

# Create a test client for sending HTTP requests to the app
client = TestClient(app)

def test_health_check():
    """
    Test the root ("/") endpoint to confirm the API is running.
    - Expects HTTP 200 response
    - Validates JSON response contains 'status' = 'online'
    - Confirms the response includes 'model' key
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "online"
    assert "model" in response.json()

@patch('api.app.model')
def test_predict_with_mock_model(mock_model):
    """
    Test the /predict endpoint with a mocked model:
    - Mocks model.predict to return a known probability
    - Sends a sample sequence and track_id as JSON
    - Asserts that the response is HTTP 200 and contains expected keys:
      'track_id', 'reentry_probability', 'prediction', and 'inference_time_ms'
    """
    mock_model.predict.return_value = np.array([[0.8]])
    
    test_data = {
        "sequence": [[1.0, 2.0, 3.0, 4.0, 5.0], [1.1, 2.1, 3.1, 4.1, 5.1]],
        "track_id": "test_track_001"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "track_id" in result
    assert "reentry_probability" in result
    assert "prediction" in result
    assert "inference_time_ms" in result
    assert result["track_id"] == "test_track_001"

def test_predict_invalid_input():
    """
    Test the /predict endpoint with invalid input:
    - Sends improperly formatted 'sequence' field
    - Expects HTTP 422 Unprocessable Entity (validation error)
    """
    invalid_data = {
        "sequence": "invalid_format",
        "track_id": "test"
    }
    
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

@patch('api.app.model')
def test_batch_predict(mock_model):
    """
    Test the /batch-predict endpoint with mocked model:
    - Simulates prediction on multiple sequences
    - Verifies response includes 'results' and correct 'batch_size'
    """
    mock_model.predict.return_value = np.array([[0.7]])
    
    batch_data = {
        "sequences": [
            {
                "sequence": [[1.0, 2.0, 3.0, 4.0, 5.0]],
                "track_id": "track_1"
            },
            {
                "sequence": [[2.0, 3.0, 4.0, 5.0, 6.0]],
                "track_id": "track_2"
            }
        ]
    }
    
    response = client.post("/batch-predict", json=batch_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "results" in result
    assert "batch_size" in result
    assert result["batch_size"] == 2
