# Purpose: Evaluate model performance and compare against baseline

import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocessing import preprocess_train_data

def main():
    """
    Main evaluation function to assess LSTM model performance against baseline.
    
    This script:
    1. Loads validation data and the trained model
    2. Generates predictions using the LSTM model
    3. Calculates standard classification metrics
    4. Compares against a simple baseline classifier
    5. Reports performance improvements from the ML approach
    """
    
    # ====== STEP 1: Load Validation Data ======
    print("📂 Loading validation data...")
    # We only need validation set for evaluation
    _, X_val, _, y_val = preprocess_train_data("data/train.csv")
    print(f"Loaded {len(X_val)} validation sequences")

    # ====== STEP 2: Load Trained Model ======
    model_path = "models/reentry_lstm.h5"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at: {model_path}")
    print("✅ Model found. Loading...")

    model = load_model(model_path)
    print(f"Model loaded: {model.name}")

    # ====== STEP 3: Make Predictions ======
    print("🔍 Making predictions...")
    # Use ragged tensors to handle variable-length sequences
    y_pred_probs = model.predict(tf.ragged.constant(X_val)).flatten()
    # Apply threshold of 0.5 to get binary predictions
    y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]
    print(f"Generated predictions for {len(y_pred)} sequences")

    # ====== STEP 4: Evaluate Model Performance ======
    # Calculate standard classification metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    # Calculate AUC only if we have both classes in validation set
    # This handles edge cases where validation might only have one class
    if len(np.unique(y_val)) == 2:
        auc = roc_auc_score(y_val, y_pred_probs)
        auc_str = f"{auc:.4f}"
    else:
        auc = None
        auc_str = "⚠️ Not computed (only one class in validation data)"

    print("\n📊 LSTM Model Performance:")
    print(f"Accuracy  : {accuracy:.4f}  # Percentage of correct predictions")
    print(f"Precision : {precision:.4f}  # Of predicted reentries, how many were correct")
    print(f"Recall    : {recall:.4f}  # Of actual reentries, how many were detected")
    print(f"F1 Score  : {f1:.4f}  # Harmonic mean of precision and recall")
    print(f"ROC AUC   : {auc_str}  # Area under ROC curve - discrimination ability")

    # ====== STEP 5: Compare with Baseline Classifier ======
    # Baseline: Classify as reentry if altitude exceeds threshold
    # This demonstrates the value of the ML approach over simple heuristics
    def baseline_predict(X, threshold=50000):
        """Simple altitude threshold classifier (baseline)"""
        return [1 if np.min(seq[:, 2]) < threshold else 0 for seq in X]

    baseline_preds = baseline_predict(X_val)

    # Calculate same metrics for baseline
    baseline_accuracy = accuracy_score(y_val, baseline_preds)
    baseline_precision = precision_score(y_val, baseline_preds)
    baseline_recall = recall_score(y_val, baseline_preds)
    baseline_f1 = f1_score(y_val, baseline_preds)

    print("\n🧪 Baseline Threshold Model Performance:")
    print(f"Accuracy  : {baseline_accuracy:.4f}")
    print(f"Precision : {baseline_precision:.4f}")
    print(f"Recall    : {baseline_recall:.4f}")
    print(f"F1 Score  : {baseline_f1:.4f}")

    # Calculate and display improvements over baseline
    print("\n📈 Improvement Over Baseline:")
    print(f"Accuracy Δ : {accuracy - baseline_accuracy:+.4f}")
    print(f"Precision Δ: {precision - baseline_precision:+.4f}")
    print(f"Recall Δ   : {recall - baseline_recall:+.4f}")
    print(f"F1 Score Δ : {f1 - baseline_f1:+.4f}")

if __name__ == "__main__":
    main()