#TODO: Make comments and clean the
#Phase two preprocessing data

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_train_data(file_path):
    """
    Preprocess training data for LSTM model.
    
    Args:
        file_path: Path to the training CSV file
        
    Returns:
        X_train, X_val, y_train, y_val: Training and validation datasets
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by track_id and timestamp
    df = df.sort_values(['track_id', 'timestamp'])
    
    # Encode categorical features
    le = LabelEncoder()
    df['sensor_id'] = le.fit_transform(df['sensor_id'])
    
    # Define features to use
    features = ['latitude', 'longitude', 'altitude', 'radiometric_intensity', 'sensor_id']
    
    # Normalize numerical features
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])
    
    # Group by track_id to create sequences
    sequences = []
    labels = []
    
    for track_id, group in df.groupby('track_id'):
        # Extract features as a sequence
        X = group[features].values
        
        # Get reentry phase label (1 if any point in sequence is reentry)
        y = int(np.any(group['reentry_phase'].values))
        
        sequences.append(X)
        labels.append(y)
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        sequences, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    return X_train, X_val, y_train, y_val

def preprocess_test_data(file_path, scaler=None):
    """
    Preprocess test data for inference.
    
    Args:
        file_path: Path to the test CSV file
        scaler: Fitted scaler from training (optional)
        
    Returns:
        X_test: Test sequences
        track_ids: List of track IDs for each sequence
    """
    # Load data
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by track_id and timestamp
    df = df.sort_values(['track_id', 'timestamp'])
    
    # Encode categorical features
    le = LabelEncoder()
    df['sensor_id'] = le.fit_transform(df['sensor_id'])
    
    # Define features to use
    features = ['latitude', 'longitude', 'altitude', 'radiometric_intensity', 'sensor_id']
    
    # Normalize numerical features
    if scaler is None:
        scaler = StandardScaler()
        df[features] = scaler.fit_transform(df[features])
    else:
        df[features] = scaler.transform(df[features])
    
    # Group by track_id to create sequences
    sequences = []
    track_ids = []
    
    for track_id, group in df.groupby('track_id'):
        # Extract features as a sequence
        X = group[features].values
        
        sequences.append(X)
        track_ids.append(track_id)
    
    return sequences, track_ids
