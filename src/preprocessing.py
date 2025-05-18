#TODO: Make comments and clean the
#Phase two preprocessing data

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_train_data(df):
    df = pd.read_csv("data/train.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['track_id', 'timestamp'])

    
    le = LabelEncoder()
    df['sensor_id'] = le.fit_transform(df['sensor_id'])

    features = ['latitude', 'longitude', 'altitude', 'radiometric_intensity', 'sensor_id']
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    sequences = []
    labels = []

    for track_id, group in df.groupby('track_id'):
        X = group[features].values
        y = group['reentry_phase'].values
        sequences.append(X)
        labels.append(int(np.any(y))) # Binary label: 1 if any timestep has reentry_phase

    X_train, X_val, y_train, y_val = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    return X_train, X_val, y_train, y_val
