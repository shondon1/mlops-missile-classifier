#TODO: Clean up the comments or add the steps to all code files

# src/evaluate.py

import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.preprocessing import preprocess_train_data

# ====== STEP 1: Load Validation Data ======
print("📂 Loading validation data...")
_, X_val, _, y_val = preprocess_train_data("data/train.csv")  # Only need validation set

# ====== STEP 2: Load Trained Model ======
model_path = "models/reentry_lstm.h5"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found at: {model_path}")
print("✅ Model found. Loading...")

model = load_model(model_path)

# ====== STEP 3: Make Predictions ======
print("🔍 Making predictions...")
y_pred_probs = model.predict(tf.ragged.constant(X_val)).flatten()
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]

# ====== STEP 4: Evaluate Metrics ======
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Safe AUC computation
if len(np.unique(y_val)) == 2:
    auc = roc_auc_score(y_val, y_pred_probs)
    auc_str = f"{auc:.4f}"
else:
    auc = None
    auc_str = "⚠️ Not computed (only one class in y_val)"

print("\n📊 LSTM Model Performance:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {auc_str}")

# ====== STEP 5: Baseline Comparison ======
def baseline_predict(X, threshold=50000):
    return [1 if max(seq[:, 2]) > threshold else 0 for seq in X]

baseline_preds = baseline_predict(X_val)

baseline_accuracy = accuracy_score(y_val, baseline_preds)
baseline_precision = precision_score(y_val, baseline_preds)
baseline_recall = recall_score(y_val, baseline_preds)
baseline_f1 = f1_score(y_val, baseline_preds)

print("\n🧪 Baseline Threshold Model Performance:")
print(f"Accuracy  : {baseline_accuracy:.4f}")
print(f"Precision : {baseline_precision:.4f}")
print(f"Recall    : {baseline_recall:.4f}")
print(f"F1 Score  : {baseline_f1:.4f}")

print("\n📈 Improvement Over Baseline:")
print(f"Accuracy Δ : {accuracy - baseline_accuracy:+.4f}")
print(f"Precision Δ: {precision - baseline_precision:+.4f}")
print(f"Recall Δ   : {recall - baseline_recall:+.4f}")
print(f"F1 Score Δ : {f1 - baseline_f1:+.4f}")
