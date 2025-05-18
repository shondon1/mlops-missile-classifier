# 1. Import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 2. Import the trained model
from tensorflow.keras.models import load_model

# Flatten ragged lists if necessary
y_val_true = [label for label in y_val]  # Already binary
# 1. Load your model
model = load_model("models/reentry_lstm.h5")

# 2. Predict on validation set
y_pred_probs = model.predict(tf.ragged.constant(X_val))

# 3. Convert probabilities to binary predictions
#Saying look if the model is 83% sure this is a reentry thats a 1 else (which is 22% sure) that it is a zero
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_probs]

# Accuracy: How many total predictions were correct
accuracy = accuracy_score(y_val_true, y_pred)

# Precision: Of the ones we predicted "reentry", how many were actually right?
precision = precision_score(y_val_true, y_pred)

# Recall: Of all the actual reentries, how many did we catch?
recall = recall_score(y_val_true, y_pred)

# F1 Score: A balance between precision and recall
f1 = f1_score(y_val_true, y_pred)

# AUC Score: How well the model separates reentry vs not reentry
auc = roc_auc_score(y_val_true, y_pred_probs)

# Print everything nicely
print("LSTM Model Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {auc:.4f}")


# Predict "1" if any altitude in sequence > THRESHOLD
def baseline_predict(X, threshold=50000):
    preds = []
    for seq in X:
        if max(seq[:, 2]) > threshold:   # HINT: 2 is the index of "altitude"
            preds.append(1)
        else:
            preds.append(0)
    return preds

baseline_preds = baseline_predict(X_val)

print("Baseline Performance:")
print("Accuracy:", accuracy_score(y_val_true, baseline_preds))
print("Precision:", precision_score(y_val_true, baseline_preds))
print("Recall:", recall_score(y_val_true, baseline_preds))
print("F1 Score:", f1_score(y_val_true, baseline_preds))
print("ROC AUC:", roc_auc_score(y_val_true, baseline_preds))
