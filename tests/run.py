
from src.preprocessing import preprocess_train_data
from models.lstm_model import build_lstm_model, train_lstm_model, save_model
from keras.models import load_model
import tensorflow as tf
from sklearn.metrics import accuracy_score
import os
import numpy as np
import tensorflow as tf
from keras.models import load_model
from sklearn.metrics import accuracy_score

from src.preprocessing import preprocess_train_data
from models.lstm_model import build_lstm_model, train_lstm_model, save_model

# I love emojis 
# ====== STEP 0: Make sure models directory exists ======
os.makedirs("models", exist_ok=True)

# ====== STEP 1: Preprocessing ======
print("🔄 Running preprocessing...")
X_train, X_val, y_train, y_val = preprocess_train_data("data/train.csv")
print(f"✅ Training sequences: {len(X_train)}")
print(f"✅ Validation sequences: {len(X_val)}")
print(f"📐 Sample shape: {X_train[0].shape}")

# ====== STEP 2: Build and Train Model ======
print("🧠 Building LSTM model...")
model = build_lstm_model((None, 5))

print("🎯 Training model...")
model, history = train_lstm_model(model, X_train, y_train, X_val, y_val, epochs=3)

# ====== STEP 3: Save Model ======
print("💾 Saving model to models/reentry_lstm.h5...")
save_model(model)

# ====== STEP 4: Load Model and Predict ======
print("📂 Loading saved model...")
model = load_model("models/reentry_lstm.h5")

print("🔍 Running predictions on validation set...")
y_pred_probs = model.predict(tf.ragged.constant(X_val))
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]

accuracy = accuracy_score(y_val, y_pred)
print(f"✅ Validation Accuracy: {accuracy:.4f}")

