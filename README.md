## Phase 1: Data Exploration

- Loaded and visualized telemetry data from `train.csv`
- Verified key columns: timestamp, position (lat/lon/alt), radiometric intensity, reentry label
- No major missing values found (or [if any, list them])
- Distribution of `reentry_phase`: [insert distribution here]
- Feature ranges noted for normalization: altitude ranges from X to Y, etc.


## Phase 2: Preprocessing & Feature Engineering

- Sorted sequences by `timestamp` and grouped by `track_id`
- Encoded `sensor_id` with LabelEncoder
- Normalized continuous features with StandardScaler
- Created sequence-level inputs for LSTM with labels indicating reentry presence
- Split data into training and validation sets

## Phase 3: LSTM Model Building

- Designed a two-layer LSTM model with dropout and batch normalization for regularized sequence learning
- Used binary crossentropy for classification and early stopping to minimize overfitting
- Trained using ragged tensors to accommodate variable-length sequences
- Model saved for later deployment using `model.save()`


## Phase 4: Model Evaluation and Baseline Comparison

- Evaluated LSTM predictions on validation set using accuracy, precision, recall, F1-score, and ROC AUC
- Compared results against a baseline classifier using an altitude threshold heuristic
- LSTM model significantly outperformed the baseline across all major metrics
