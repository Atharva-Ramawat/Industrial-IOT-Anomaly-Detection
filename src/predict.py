import numpy as np
import tensorflow as tf

def detect_anomalies(model, new_data_sequences, threshold):
    """
    Passes data through the autoencoder. If the reconstruction error
    is higher than the threshold, it flags it as an anomaly.
    """
    # Get the model's attempt to reconstruct the data
    reconstructions = model.predict(new_data_sequences)
    
    # Calculate Mean Squared Error between original and reconstruction
    mse = np.mean(np.power(new_data_sequences - reconstructions, 2), axis=(1, 2))
    
    # Flag anomalies
    anomalies = mse > threshold
    return anomalies, mse