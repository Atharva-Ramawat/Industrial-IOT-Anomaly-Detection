import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed

def build_lstm_autoencoder(sequence_length, num_features):
    """
    Builds an LSTM Autoencoder for anomaly detection.
    """
    model = Sequential([
        # --- ENCODER ---
        # Reads the sequence and compresses it down to the most important patterns
        LSTM(64, activation='relu', input_shape=(sequence_length, num_features), return_sequences=True),
        LSTM(32, activation='relu', return_sequences=False),
        
        # --- BOTTLENECK ---
        # Prepares the compressed data to be reconstructed
        RepeatVector(sequence_length),
        
        # --- DECODER ---
        # Tries to recreate the original input sequence from the compressed version
        LSTM(32, activation='relu', return_sequences=True),
        LSTM(64, activation='relu', return_sequences=True),
        
        # Output layer matches the exact shape of the input data
        TimeDistributed(Dense(num_features))
    ])
    
    # We use Mean Squared Error (MSE) to measure how badly the model reconstructs the data.
    # High MSE = Anomaly!
    model.compile(optimizer='adam', loss='mse')
    
    return model

if __name__ == "__main__":
    # Quick test to see if it compiles and view the architecture
    dummy_model = build_lstm_autoencoder(sequence_length=50, num_features=15)
    dummy_model.summary()