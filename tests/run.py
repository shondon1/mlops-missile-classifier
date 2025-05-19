# Purpose: Main script to preprocess data, train model, and validate results

import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import accuracy_score

from src.preprocessing import preprocess_train_data
from models.lstm_model import build_lstm_model, train_lstm_model, save_model

def main():
    """
    Main entry point for the missile reentry classifier pipeline.
    
    This script orchestrates the end-to-end ML workflow:
    1. Loads and preprocesses telemetry data
    2. Builds and trains an LSTM model for sequence classification
    3. Saves the trained model for deployment
    4. Validates the saved model works as expected
    
    The workflow demonstrates a complete ML pipeline from data to
    validated model ready for production deployment.
    """
    
    # ====== STEP 0: Ensure directory structure exists ======
    os.makedirs("models", exist_ok=True)
    print("🔧 Environment prepared")

    # ====== STEP 1: Data Preprocessing ======
    print("🔄 Preprocessing telemetry data...")
    X_train, X_val, y_train, y_val = preprocess_train_data("data/train.csv")
    print(f"✅ Training sequences: {len(X_train)}")
    print(f"✅ Validation sequences: {len(X_val)}")
    print(f"📐 Feature dimension: {X_train[0].shape[1]}")
    
    # Log class distribution to monitor for imbalance
    train_class_dist = np.bincount(y_train)
    val_class_dist = np.bincount(y_val)
    print(f"Training class distribution: {train_class_dist}")
    print(f"Validation class distribution: {val_class_dist}")

    # ====== STEP 2: Model Building and Training ======
    print("🧠 Building LSTM sequence classifier...")
    # Input shape: (None, 5) represents variable sequence length with 5 features
    model = build_lstm_model((None, 5))
    model.summary()

    print("🎯 Training model...")
    # Set epochs higher for production model (20-50)
    # Using 3 here for demonstration purposes
    model, history = train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=3)

    # ====== STEP 3: Model Persistence ======
    print("💾 Saving model for production deployment...")
    save_model(model)
    print("✅ Model saved to models/reentry_lstm.h5")

    # ====== STEP 4: Model Validation ======
    # Reload model to verify serialization worked correctly
    print("📂 Loading model to verify persistence...")
    loaded_model = load_model("models/reentry_lstm.h5")

    print("🔍 Validating model performance...")
    y_pred_probs = loaded_model.predict(tf.ragged.constant(X_val))
    y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]

    # Calculate accuracy as quick validation metric
    accuracy = accuracy_score(y_val, y_pred)
    print(f"✅ Validation Accuracy: {accuracy:.4f}")
    print(f"🏁 Training and validation complete!")
    
    # For production, full evaluation metrics would be calculated
    # See evaluate.py for comprehensive metrics analysis

if __name__ == "__main__":
    main()