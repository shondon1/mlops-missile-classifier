# Purpose: Transform raw telemetry data into model-ready sequences

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_train_data(file_path):
    """
    Preprocess training data for LSTM model by creating normalized sequence inputs.
    
    This function:
    1. Loads and cleans the raw telemetry data
    2. Creates sequences grouped by track_id for sequence learning
    3. Normalizes numerical features for better model convergence
    4. Transforms categorical variables into numerical representations
    5. Splits data into training and validation sets with stratification
    
    Args:
        file_path (str): Path to the training CSV file
        
    Returns:
        tuple: (X_train, X_val, y_train, y_val) containing:
            - X_train: List of sequence arrays for training
            - X_val: List of sequence arrays for validation
            - y_train: Binary labels for training (1=reentry, 0=non-reentry)
            - y_val: Binary labels for validation
    """
    # Load raw telemetry data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime for proper sequence ordering
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by track_id and timestamp to maintain temporal order within tracks
    df = df.sort_values(['track_id', 'timestamp'])
    
    # Encode categorical sensor_id to numerical values for model input
    le = LabelEncoder()
    df['sensor_id'] = le.fit_transform(df['sensor_id'])
    
    # Select features for model training
    features = ['latitude', 'longitude', 'altitude', 'radiometric_intensity', 'sensor_id']
    
    # Normalize features to improve training stability and convergence
    # Zero mean and unit variance standardization
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Group telemetry points by missile track to create sequences
    sequences = []
    labels = []
    
    for track_id, group in df.groupby('track_id'):
        # Extract features as a temporal sequence
        X = group[features].values
        
        # Create binary label: 1 if any point in sequence has reentry_phase
        # This transforms the point-wise labels to sequence-level labels
        y = int(np.any(group['reentry_phase'].values))
        
        sequences.append(X)
        labels.append(y)
    
    # Split into training and validation sets
    # stratify=labels ensures balanced class distribution in both sets
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return X_train, X_val, y_train, y_val

def preprocess_test_data(file_path, scaler=None):
    """
    Preprocess test data for inference, maintaining consistency with training preprocessing.
    
    This function handles unlabeled test data and maintains the same preprocessing steps
    as the training data to ensure consistent model inputs.
    
    Args:
        file_path (str): Path to the test CSV file
        scaler (StandardScaler, optional): Fitted scaler from training for consistent normalization
        
    Returns:
        tuple: (X_test, track_ids) containing:
            - X_test: List of preprocessed sequence arrays
            - track_ids: List of corresponding track identifiers
    """
    # Load test data
    df = pd.read_csv(file_path)
    
    # Apply same preprocessing steps as training for consistency
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['track_id', 'timestamp'])
    
    # Encode categorical features
    le = LabelEncoder()
    df['sensor_id'] = le.fit_transform(df['sensor_id'])
    
    # Define the same feature set as used in training
    features = ['latitude', 'longitude', 'altitude', 'radiometric_intensity', 'sensor_id']
    
    # Use provided scaler if available, otherwise fit a new one
    # In production, always use the scaler from training
    if scaler is None:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
    else:
        df[features] = scaler.transform(df[features])
    
    # Create test sequences and track tracking identifiers
    sequences = []
    track_ids = []
    
    for track_id, group in df.groupby('track_id'):
        X = group[features].values
        sequences.append(X)
        track_ids.append(track_id)
    
    return sequences, track_ids