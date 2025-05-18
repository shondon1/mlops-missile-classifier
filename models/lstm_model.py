import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_model(input_shape: tuple) -> tf.keras.Model:
    """
    Builds a sequential LSTM model for binary classification of missile reentry phase.

    The model includes dropout for regularization and batch normalization for stable training.
    """

    model = Sequential()
    
    # First LSTM layer with return_sequences=True to stack another LSTM
    model.add(LSTM(64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))  # Helps prevent overfitting
    model.add(BatchNormalization())  # Speeds up and stabilizes training
    
    # Second LSTM layer reduces sequence to vector
    model.add(LSTM(32))
    model.add(Dropout(0.3))
    
    # Dense layer before binary output
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Sigmoid for binary classification

    # Compile the model using binary crossentropy and accuracy metric
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_lstm_model(
    model: tf.keras.Model,
    X_train: list,
    y_train: list,
    X_val: list,
    y_val: list,
    epochs: int = 20
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Trains the LSTM model using ragged tensors for variable sequence lengths.
    
    Includes early stopping to prevent overfitting and reduce unnecessary compute time.
    """
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        tf.ragged.constant(X_train), y_train,
        validation_data=(tf.ragged.constant(X_val), y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=2
    )

    return model, history

def save_model(model: tf.keras.Model, path: str = "models/reentry_lstm.h5") -> None:
    """
    Saves the trained model to disk in HDF5 format for later inference deployment.
    """
    model.save(path)
