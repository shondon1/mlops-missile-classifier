import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import numpy as np
import sys
import os

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.app import app

client = TestClient(app)

def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["status"] == "online"

@patch('api.app.model')
def test_predict_endpoint(mock_model):
    """Test prediction endpoint with mocked model"""
    mock_model.predict.return_value = np.array([[0.8]])
    
    test_data = {
        "sequence": [[1.0, 2.0, 3.0, 4.0, 5.0]],
        "track_id": "test_track"
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    result = response.json()
    assert "reentry_probability" in result
    assert "track_id" in result

def test_predict_no_model():
    """Test prediction when model is not loaded"""
    with patch('api.app.model', None):
        test_data = {
            "sequence": [[1.0, 2.0, 3.0, 4.0, 5.0]],
            "track_id": "test"
        }
        response = client.post("/predict", json=test_data)
        assert response.status_code == 500